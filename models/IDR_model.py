import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .base_model import BaseModel
from models.loss import CharbonnierLoss, VGGLoss, WeightedTVLoss, EdgeLoss, SimDLoss, IMP_Loss, OHCeLoss
import torch.nn.functional as F
import torch.distributed as dist
from metrics.psnr_ssim import psnr_torch
from timm.models.layers import to_2tuple
from models.archs import define_network
from utils.data_util import Mixing_Augment
import numpy as np

logger = logging.getLogger('base')


class IDR_Model(BaseModel):
    def __init__(self, opt, args):
        super(IDR_Model, self).__init__(opt)
        # opt

        self.opt = opt
        self.local_rank = args.local_rank  # non dist training
        self.nprocs = args.nprocs
        train_opt = opt['train']

        # if mixing lq and gt
        self.mixing_flag = train_opt['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = train_opt['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = train_opt['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.local_rank)

        # define network and load pretrained models
        self.net_G = define_network(opt['network_g']).cuda(self.local_rank)

        if opt['dist']:
            self.net_G = DistributedDataParallel(self.net_G, device_ids=[self.local_rank])
        else:
            self.net_G = self.net_G

        # print network
        self.print_network()
        if opt['pretrained']:
            self.load()

        ####################################################################
        if self.is_train:
            self.net_G.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().cuda(self.local_rank)
            elif loss_type == 'IDR':
                self.cri_pix = nn.L1Loss().cuda(self.local_rank)
                self.cri_pred = OHCeLoss().cuda(self.local_rank)
                self.weight1 = train_opt['loss_1_weight']
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            # set up optimizers and schedulers
            self.setup_optimizers()
            if train_opt['Auto_scheduler']:
                self.setup_schedulers()

            self.log_dict = OrderedDict()

    def feed_train_data(self, data, need_GT=True):
        self.lq = data['degrad_patch'].cuda(self.local_rank, non_blocking=True)
        self.degra = data['degra']
        self.de_label = data['de_label'].cuda(self.local_rank, non_blocking=True)
        if need_GT:
            self.gt = data['clean_patch'].cuda(self.local_rank, non_blocking=True)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
    
    def feed_data(self, data, need_GT=True):
        self.lq = data['lq'].cuda(self.local_rank, non_blocking=True)
        self.de_type = data['degra']
        self.de_label = data['de_label'].cuda(self.local_rank, non_blocking=True)
        if need_GT:
            self.gt = data['gt'].cuda(self.local_rank, non_blocking=True)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def reduce_mean(self, tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def optimize_parameters(self, step, epoch):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.net_G.zero_grad() 
        self.optimizer_G.zero_grad()

        output = self.net_G(inp_img=self.lq, degra_type=self.degra, gt=self.gt, epoch=epoch)
        if type(output) == list:
            self.output = output[0]
        else:
            self.output = output


        if self.loss_type =='l1':
            l_pix = self.cri_pix(self.output, self.gt)
            l_total = l_pix
        elif self.loss_type == "IDR":
            l_pix = self.cri_pix(self.output, self.gt) + output[1]
            l_total = l_pix
            if epoch <= self.opt['train']['epochs_encoder']:
                l_pred = self.weight1*np.sum([self.cri_pred(output[j],self.de_label) for j in range(2,len(output))])
                l_total += l_pred

        psnr = psnr_torch(self.output.detach(), self.gt.detach())

        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), 0.01)
        self.optimizer_G.step()
        
        if self.opt['dist']:
            torch.distributed.barrier()
            l_total = self.reduce_mean(l_total, self.nprocs)
            psnr = self.reduce_mean(psnr, self.nprocs)

        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_pred'] = l_pred.item() if epoch <= self.opt['train']['epochs_encoder'] else 0

    def test(self,epoch):
        self.net_G.eval()
        with torch.no_grad():
            out = self.net_G(self.lq,degra_type=self.de_type,epoch=epoch)
            if type(out) == list:
                self.output = out[0]
            else:
                self.output = out
        self.net_G.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.lq
        out_dict['rlt'] = self.output
        if need_GT:
            out_dict['GT'] = self.gt
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net_G)
        if isinstance(self.net_G, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.net_G.__class__.__name__,
                                             self.net_G.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net_G.__class__.__name__)
        if self.local_rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading models for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.net_G, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.net_G, 'G', iter_label)

    def save_best(self, name):
        self.save_network(self.net_G, 'best' + name, 0)

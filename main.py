import os
import math
import argparse
import random
import logging

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

import utils.option as option
import utils.util as util
import utils.data_util as udata
from utils.metric_util import AverageMeter
from metrics.psnr_ssim import calculate_psnr
from data import create_dataloader, create_dataset
from models import create_model
from models.lr_scheduler import get_position_from_periods
from metrics.psnr_ssim import compute_psnr_ssim, calculate_psnr, calculate_ssim
from utils.tensor_op import patchifyBCPHW


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./config/IDR-restormer.yml',
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')  
    args = parser.parse_args()
    opt = option.parse(args.opt)
    train_opt = opt['train']

    #### distributed training settings
    args.nprocs = torch.cuda.device_count()  # dist.get_world_size()
    if opt['dist']:
        # rank = int(os.environ["LOCAL_RANK"])
        # args.local_rank = rank
        rank = args.local_rank    
        dist.init_process_group(backend='nccl', )
        torch.cuda.set_device(rank)
        print('Enabled distributed training.')
    else:
        rank = 0
        args.local_rank = 0   # point the specific gpu
        args.nprocs = 1

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        # device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda())
        # option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['logger']['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.2:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(os.path.join(opt['path']['root'], 'tb_logger', opt['name']))

    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt) 
            # total_iters = int(opt['train']['niter'])  # Total number of iterations
            train_epochs = opt['train']['epoch']
            epochs_encoder = opt['train']['epochs_encoder']
            # number of iterations required to traverse the dataset
            iterations_per_epoch = int(math.ceil(len(train_set) / dataset_opt['batch_size_per_gpu'] / args.nprocs))
            total_iters = iterations_per_epoch * train_epochs
            

            # 使用 DistributedSampler 对数据集进行划分
            if opt['dist']:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), iterations_per_epoch))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    train_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            if opt['dist']:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
            else:
                val_sampler = None
            val_loader = create_dataloader(val_set, dataset_opt, val_sampler)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    assert train_loader is not None

    #### create models
    model = create_model(opt, args)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_iter = 0
        start_epoch = 0

    best_psnr_avg = 0
    best_step_avg = 0
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_iter))

    # progress training
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    patch_size = opt['datasets']['train'].get('patch_size')
    per_batch_sizes = opt['datasets']['train'].get('per_batch_sizes')
    per_patch_sizes = opt['datasets']['train'].get('per_patch_sizes')
    milestone = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))]) 
    logger_change = [True] * len(milestone)


    torch.autograd.set_detect_anomaly(True)
    try:
        for epoch in range(start_epoch, train_epochs + 2):
            # 
            mean_psnr = AverageMeter()
            total_loss = AverageMeter()
            pix_loss = AverageMeter()
            pred_loss = AverageMeter()
            heo_loss = AverageMeter()
            uni_loss = AverageMeter()
            imp_loss = AverageMeter()

            print_iter = 0

            if epoch == epochs_encoder+1:
                logger.info('Saving first stage models and training states.')
                model.save(epoch)
                model.save_training_state(epoch, current_iter)
                train_set.set_per_degra(1)
                train_loader = create_dataloader(train_set, opt['datasets']['train'], train_sampler)
                logger.info('Successfully initializing the second stage Dataloader.')

            for batch_step, train_data in enumerate(train_loader):
                current_iter += 1
                if current_iter > total_iters:
                    break

                ### ------Progressive learning ---------------------  (manner: loader data)
                if opt['datasets']['train']['progressive_training']:
                    pos = get_position_from_periods(current_iter, milestone)

                    per_batch_size = per_batch_sizes[pos]
                    per_patch_size = per_patch_sizes[pos]

                    if logger_change[pos]:
                        logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(per_patch_size,
                                                                                                per_batch_size * torch.cuda.device_count()))
                        logger_change[pos] = False

                    # get mini batch
                    if per_batch_size < batch_size:
                        train_data['lq'], train_data['gt'] = udata.get_tensor_batch(train_data['lq'], train_data['gt'],
                                                                                    per_batch_size)

                    # get mini patch
                    if per_patch_size < patch_size:
                        train_data['lq'], train_data['gt'] = udata.get_tensor_patch(train_data['lq'], train_data['gt'],
                                                                                    per_patch_size)
                ###-------------------------------------------

                #### training
                model.feed_train_data(train_data)
                
                #### optimizer update
                model.optimize_parameters(current_iter, epoch)

                #### update learning rate
                if train_opt['Auto_scheduler']:
                    model.update_learning_rate(current_iter, warmup_iter=opt['train']['warmup_iter'])
                else:
                    if epoch <= epochs_encoder:
                        lr = train_opt['optim_g']['lr'] * (0.1 ** (epoch // 60))
                        model._set_lr([[lr]])
                    else:
                        lr = 0.0001 * (0.5 ** ((epoch - opt['train']['epochs_encoder']) // 125))
                        model._set_lr([[lr]])


                #### log
                if current_iter % opt['logger']['print_freq'] == 0:
                    print_iter += 1  ############################################################## new
                    logs = model.get_current_log()
                    message = '[epoch:{:3d}, iter:{:4,d}/{:8,d}, lr:('.format(epoch, current_iter % iterations_per_epoch,
                                                                            current_iter)
                    for v in model.get_current_learning_rate():
                        message += '{:.3e},'.format(v)
                    message += ')] '
                    
                    pix_loss.update(logs['l_pix'], print_iter)
                    pred_loss.update(logs['l_pred'], print_iter)
                    mean_psnr.update(logs['psnr'], print_iter)

                    message += '{:s}: {:.4e} '.format('l_pix', pix_loss.avg)
                    message += '{:s}: {:.4e} '.format('l_pred', pred_loss.avg)
                    message += '{:s}: {:} '.format('mean_psnr', mean_psnr.avg)

                    # tensorboard logger
                    if opt['logger']['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar('mean_psnr', mean_psnr, current_iter)
                    if rank <= 0:
                        logger.info(message)

            ##### valid test
            if opt.get('val', None) and ((epoch < opt['val']['val_milelstone'] and epoch % opt['val']['val_epoch'][0] == 0) or (epoch >= opt['val']['val_milelstone'] and epoch % opt['val']['val_epoch'][1] == 0) or epoch==451):
                avg_rain_psnr = AverageMeter()
                avg_haze_psnr = AverageMeter()
                avg_noise_psnr = AverageMeter()
                avg_blur_psnr = AverageMeter()
                avg_light_psnr = AverageMeter()


                avg_rain_ssim = AverageMeter()
                avg_haze_ssim = AverageMeter()
                avg_noise_ssim = AverageMeter()
                avg_blur_ssim = AverageMeter()
                avg_light_ssim = AverageMeter()


                avg_psnr = AverageMeter()

                for val_data in val_loader:

                    img_name = val_data['lq_path'][0].split('/')[-1].split('.')[0]


                    model.feed_data(val_data)
                    model.test(epoch)

                    visuals = model.get_current_visuals()
                    rlt_img = visuals['rlt']
                    gt_img = visuals['GT']

                    # Save Restored images for reference
                    if opt['val']['save_img'] and epoch % opt['val']['save_image_epoch'] == 0:
                        img_dir = os.path.join(opt['path']['val_images'], str(epoch))
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                        util.save_img(rlt_img, save_img_path)
                    

                    psnr_inst,ssim_inst,N = compute_psnr_ssim(rlt_img, gt_img)
                    if math.isinf(psnr_inst) or math.isnan(psnr_inst):
                        psnr_inst = 0


                    if 'derain' in val_data['lq_path'][0]:
                        avg_rain_psnr.update(psnr_inst,N)
                        avg_rain_ssim.update(ssim_inst,N)
                        avg_psnr.update(psnr_inst,N)
                    elif 'dehaze' in val_data['lq_path'][0]:
                        avg_haze_psnr.update(psnr_inst,N)
                        avg_haze_ssim.update(ssim_inst,N)
                        avg_psnr.update(psnr_inst,N)
                    elif 'denoise' in val_data['lq_path'][0]:
                        avg_noise_psnr.update(psnr_inst,N)
                        avg_noise_ssim.update(ssim_inst,N)
                        avg_psnr.update(psnr_inst,N)
                    elif 'deblur' in val_data['lq_path'][0]:
                        avg_blur_psnr.update(psnr_inst,N)
                        avg_blur_ssim.update(ssim_inst,N)
                        avg_psnr.update(psnr_inst,N)
                    elif 'delowlight' in val_data['lq_path'][0]:
                        avg_light_psnr.update(psnr_inst,N)
                        avg_light_ssim.update(ssim_inst,N)
                        avg_psnr.update(psnr_inst,N)


                    if (avg_psnr.count) % 100 == 0:
                        logger.info("Test the {} the image".format(avg_psnr.count))

                # log
                if opt['dist']:
                    avg_rain_psnr.avg = util.reduce_mean(torch.tensor(avg_rain_psnr.avg).cuda(rank), args.nprocs)
                    avg_haze_psnr.avg = util.reduce_mean(torch.tensor(avg_haze_psnr.avg).cuda(rank), args.nprocs)
                    avg_noise_psnr.avg = util.reduce_mean(torch.tensor(avg_noise_psnr.avg).cuda(rank), args.nprocs)
                    avg_blur_psnr.avg = util.reduce_mean(torch.tensor(avg_blur_psnr.avg).cuda(rank), args.nprocs)
                    avg_light_psnr.avg = util.reduce_mean(torch.tensor(avg_light_psnr.avg).cuda(rank), args.nprocs)

                    avg_rain_ssim.avg = util.reduce_mean(torch.tensor(avg_rain_ssim.avg).cuda(rank), args.nprocs)
                    avg_haze_ssim.avg = util.reduce_mean(torch.tensor(avg_haze_ssim.avg).cuda(rank), args.nprocs)
                    avg_noise_ssim.avg = util.reduce_mean(torch.tensor(avg_noise_ssim.avg).cuda(rank), args.nprocs)
                    avg_blur_ssim.avg = util.reduce_mean(torch.tensor(avg_blur_ssim.avg).cuda(rank), args.nprocs)
                    avg_light_ssim.avg = util.reduce_mean(torch.tensor(avg_light_ssim.avg).cuda(rank), args.nprocs)
                    avg_psnr.avg = util.reduce_mean(torch.tensor(avg_psnr.avg).cuda(rank), args.nprocs)

                logger.info(
                    '# Validation # Average Rain PSNR: {:.4e} Average Haze PSNR: {:.4e} Average Noise PSNR: {:.4e} Average Blur PSNR: {:.4e} Average Light PSNR: {:.4e}'.
                        format(avg_rain_psnr.avg, avg_haze_psnr.avg, avg_noise_psnr.avg, avg_blur_psnr.avg, avg_light_psnr.avg))
                logger.info(
                    '# Validation # Average Rain SSIM: {:.4e} Average Haze SSIM: {:.4e} Average Noise SSIM: {:.4e} Average Blur SSIM: {:.4e} Average Light SSIM: {:.4e}'.
                        format(avg_rain_ssim.avg, avg_haze_ssim.avg, avg_noise_ssim.avg, avg_blur_ssim.avg, avg_light_ssim.avg))

                logger.info(
                    '# Validation # Average PSNR: {:.4e} Previous best Average PSNR: {:.4e} Previous best Average step: {}'.
                        format(avg_psnr.avg, best_psnr_avg, best_step_avg))

                if avg_psnr.avg > best_psnr_avg:
                    if rank <= 0:
                        best_psnr_avg = avg_psnr.avg
                        best_step_avg = current_iter
                        logger.info('Saving best average models!!!!!!!The best psnr is:{:4e}'.format(best_psnr_avg))
                        model.save_best('avg_' + str(np.round(best_psnr_avg, 4)))
                        model.save_training_state(epoch, current_iter,best=True)

            #### save models and training states
            if epoch % opt['logger']['save_checkpoint_epoch'] == 0 and epoch >= 200:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch)
                    model.save_training_state(epoch, current_iter)

        if rank <= 0:
            logger.info('Saving the final models.')
            model.save('latest')
            logger.info('End of training.')
            # tb_logger.close()

    except:
        if opt['interupt_protection']:
            logger.info("-" * 89)
            logger.info("Exiting from training early. Saving models")
            model.save(epoch)
            model.save_training_state(epoch, current_iter)


if __name__ == '__main__':
    main()

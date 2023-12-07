import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import logging

from utils.metric_util import AverageMeter
from utils.tensor_op import save_img_tensor, save_image_tensor
from utils.util import mkdir, setup_logger
from utils.data_util import crop_HWC_img, random_augmentation, tensor2img
from metrics.psnr_ssim import compute_psnr_ssim, calculate_psnr, calculate_ssim

from models.archs.IDR_restormer_arch import IDR_restormer



class DenoiseTestDataset(Dataset):
    def __init__(self, args, dataset="CBSD68"):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15
        self.dataset_dict = {'CBSD68': 0, 'urban100': 1, 'Kodak24':2}

        self.set_dataset(dataset)

        self.toTensor = ToTensor()

    def _init_clean_ids(self):

        if self.task_idx == 0:
            self.clean_ids = []
            name_list = os.listdir(self.args.denoise_CBSD68_path)
            self.clean_ids += [self.args.denoise_CBSD68_path + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.clean_ids = []
            name_list = os.listdir(self.args.denoise_urban100_path)
            self.clean_ids += [self.args.denoise_urban100_path + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.clean_ids = []
            name_list = os.listdir(self.args.denoise_Kodak24_path)
            self.clean_ids += [self.args.denoise_Kodak24_path + id_ for id_ in name_list]
        self.num_clean = len(self.clean_ids)

    def set_dataset(self, dataset):
        self.task_idx = self.dataset_dict[dataset]
        self._init_clean_ids()

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
    def _edgeComputation(self,x):
        x_diffx = np.abs(x[:,1:,:] - x[:,:-1,:])
        x_diffy = np.abs(x[1:,:,:] - x[:-1,:,:])

        y = np.zeros_like(x)
        y[:,1:,:] += x_diffx
        y[:,:-1,:] += x_diffx
        y[1:,:,:] += x_diffy
        y[:-1,:,:] += x_diffy
        y = np.sum(y,2)/3
        y /= 4
        return y[:,:,None].astype(np.float32)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_HWC_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=32)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur':2, 'low-light':3, 'UDC_T':4, 'UDC_P':5}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.deblur_path + 'input/')
            self.ids += [self.args.deblur_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.low_light_path + 'input/')
            self.ids += [self.args.low_light_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 4:
            self.ids = []
            name_list = os.listdir(self.args.udc_T_path + 'input/')
            self.ids += [self.args.udc_T_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 5:
            self.ids = []
            name_list = os.listdir(self.args.udc_P_path + 'input/')
            self.ids += [self.args.udc_P_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 4:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 5:
            gt_name = degraded_name.replace("input", "target")
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def _edgeComputation(self,x):
        x_diffx = np.abs(x[:,1:,:] - x[:,:-1,:])
        x_diffy = np.abs(x[1:,:,:] - x[:-1,:,:])

        y = np.zeros_like(x)
        y[:,1:,:] += x_diffx
        y[:,:-1,:] += x_diffx
        y[1:,:,:] += x_diffy
        y[:-1,:,:] += x_diffy
        y = np.sum(y,2)/3
        y /= 4
        return y[:,:,None].astype(np.float32)

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_HWC_img(np.array(Image.open(degraded_path).convert('RGB')), base=32)
        clean_img = crop_HWC_img(np.array(Image.open(clean_path).convert('RGB')), base=32)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


def test_Denoise(net, dataset, task="CBSD68", sigma=15,save_img=True):
    logger = logging.getLogger('base')
    output_path = opt.output_path + 'denoise/' + str(sigma) + '/'
    # subprocess.check_output(['mkdir', '-p', output_path])
    mkdir(output_path)
    
    dataset.set_dataset(task)
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            if type(restored) == list:
                restored = restored[0]
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            if save_img:
                save_image_tensor(restored, output_path + clean_name[0] + '.png')

        logger.info("Deonise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain",save_img=True):
    logger = logging.getLogger('base')
    output_path = opt.output_path + task + '/'
    # subprocess.check_output(['mkdir', '-p', output_path])
    mkdir(output_path)

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            if type(restored) == list:
                restored = restored[0]
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            N = degrad_patch.shape[0]
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            if save_img:
                save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        logger.info("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for 5 tasks, 1 for denoising details, 2 for unknowing UDC')

    parser.add_argument('--denoise_CBSD68_path', type=str, default="", help='save path of test noisy images')
    parser.add_argument('--denoise_urban100_path', type=str, default="", help='save path of test noisy images')
    parser.add_argument('--denoise_Kodak24_path', type=str, default="", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default="", help='save path of test blur images')
    parser.add_argument('--low_light_path', type=str, default="", help='save path of test low-light images')
    parser.add_argument('--udc_T_path', type=str, default="", help='save path of test udc Toled images')
    parser.add_argument('--udc_P_path', type=str, default="", help='save path of test udc Poled images')
    parser.add_argument('--output_path', type=str, default="./results/visualization", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="", help='checkpoint save path')
    parser.add_argument('--log_path', type=str, default="./results/log", help='checkpoint save path')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    denoise_set = DenoiseTestDataset(opt)
    derain_set = DerainDehazeDataset(opt)

    # Make network
    net = IDR_restormer(inp_channels=3, out_channels=3, dim=24, num_blocks=[2,3,3,4], num_refinement_blocks=2, heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_degra_queries = 24, keep_degra=48)

    net = net.cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.ckpt_path, map_location=torch.device(opt.cuda)))

    setup_logger('base', opt.log_path, level=logging.INFO, phase='test', screen=True, tofile=False)
    logger = logging.getLogger('base')

    if opt.mode == 0:
        logger.info('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_set, task="derain")

        logger.info('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

        logger.info('Start testing CBSD68 Sigma=25...')
        test_Denoise(net, denoise_set, task="CBSD68", sigma=25)

        logger.info('Start testing GoPro...')
        test_Derain_Dehaze(net, derain_set, task="deblur")

        logger.info('Start testing LOL...')
        test_Derain_Dehaze(net, derain_set, task="low-light")

    elif opt.mode == 1:
        logger.info('Start testing CBSD68 Sigma=15...')
        test_Denoise(net, denoise_set, task="CBSD68", sigma=15)

        logger.info('Start testing CBSD68 Sigma=25...')
        test_Denoise(net, denoise_set, task="CBSD68", sigma=25)

        logger.info('Start testing CBSD68 Sigma=50...')
        test_Denoise(net, denoise_set, task="CBSD68", sigma=50)


        logger.info('Start testing urban100 Sigma=15...')
        test_Denoise(net, denoise_set, task="urban100", sigma=15)

        logger.info('Start testing urban100 Sigma=25...')
        test_Denoise(net, denoise_set, task="urban100", sigma=25)

        logger.info('Start testing urban100 Sigma=50...')
        test_Denoise(net, denoise_set, task="urban100", sigma=50)


        logger.info('Start testing Kodak24 Sigma=15...')
        test_Denoise(net, denoise_set, task="Kodak24", sigma=15)

        logger.info('Start testing Kodak24 Sigma=25...')
        test_Denoise(net, denoise_set, task="Kodak24", sigma=25)

        logger.info('Start testing Kodak24Sigma=50...')
        test_Denoise(net, denoise_set, task="Kodak24", sigma=50)

    elif opt.mode == 2:
        logger.info('Start testing UDC_TOLED...')
        test_Derain_Dehaze(net, derain_set, task="UDC_T")

        logger.info('Start testing UDC_POLED...')
        test_Derain_Dehaze(net, derain_set, task="UDC_P")

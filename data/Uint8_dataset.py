import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
# from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np
import utils.data_util as udata
from torchvision.transforms.functional import normalize
from PIL import Image
from utils.data_util import crop_HWC_img, onehot, smooth_one_hot


def get_image_uint8(img):
    '''
    :type img : str
    :param img: img path(LDR img)

    :return: img in np.uint8 with range(0-255), The width and height are multiples of base
    '''
    img = crop_HWC_img(np.array(Image.open(img).convert('RGB')), base=32)
    return img


def load_image_train(group):
    '''
    :type group: list,len=2
    :param group: Path of image pair

    :return:img in np.float32 with range(0-1), The width and height are multiples of 4
    '''
    inputs = get_image_uint8(group[0])
    target = get_image_uint8(group[1])
    return inputs, target


def transform():
    return Compose([
        ToTensor(),
    ])


class Uint8_dataset(data.Dataset):
    '''
    :param upscale_factor:
    :param data_augmentation:
    :param group_file:
    :param patch_size:
    :param black_edges_crop:
    :param hflip:
    :param rot:
    :param transform:
    '''
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, dataset_opt):
        super(Uint8_dataset, self).__init__()

        groups = []
        self.group_file = dataset_opt['group_file']
        for i in range(len(self.group_file)):
            groups += [line.rstrip() for line in open(self.group_file[i])]

        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = dataset_opt.get('scale', 1)
        self.transform = True
        self.data_augmentation = dataset_opt.get('data_augmentation', None)
        self.patch_size = dataset_opt.get('patch_size', None)
        self.toTensor = ToTensor()
        self.sigma = 15

        self.de_label = {'denoise': 0, 'derain': 1, 'dehaze': 2, 'deblur': 3, 'delowlight': 4, 'clean': 5}
        self.classes = dataset_opt['num_degra']

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def _float_value(self, img):
        return img.astype(np.float32)/255.0

    def __getitem__(self, index):
        
        inputs, target = load_image_train(self.image_filenames[index])

        if 'urban' in self.image_filenames[index][0] or 'deblur' in self.image_filenames[index][0]:
            inputs, target = udata.get_HWC_patch(inputs, target, 512, self.upscale_factor)
            
        if self.data_augmentation == 1:
            inputs, target = udata.augment(inputs, target)
        elif self.data_augmentation == 2:
            inputs, target = udata.random_augmentation(inputs, target)
        
        if 'noise' in self.image_filenames[index][0]:
            inputs = self._add_gaussian_noise(target)

        if self.transform:
            inputs, target = self.toTensor(inputs), self.toTensor(target)

        de_type = self.image_filenames[index][0].split("/")[4]
        de_label = onehot(self.de_label[de_type],classes=self.classes)
    

        return {'lq': inputs, 'gt': target, 'lq_path': self.image_filenames[index][0], 'gt_path': self.image_filenames[index][1],'degra':de_type,'de_label':de_label}

    def __len__(self):
        return len(self.image_filenames)


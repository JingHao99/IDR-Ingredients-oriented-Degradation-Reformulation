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


def get_image_hdr(img):
    '''
    :type img : str
    :param img: img path(HDR img)

    :return: img in np.float32 with range(0-1), The width and height are multiples of 4
    '''
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)/65535.0

    w, h = img.shape[0], img.shape[1]
    while w % 4 != 0:
        w += 1
    while h % 4 != 0:
        h += 1
    img = cv2.resize(img, (h, w))

    return img

def get_image_ldr(img):
    '''
    :type img : str
    :param img: img path(LDR img)

    :return: img in np.float32 with range(0-1), The width and height are multiples of 4
    '''
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    w,h = img.shape[0],img.shape[1]
    while w%4!=0:
        w+=1
    while h%4!=0:
        h+=1
    img = cv2.resize(img,(h,w))
    return img


def load_image_train(group):
    '''
    :type group: list,len=2
    :param group: Path of image pair

    :return:img in np.float32 with range(0-1), The width and height are multiples of 4
    '''
    inputs = get_image_ldr(group[0])
    target = get_image_ldr(group[1])
    return inputs, target


def transform():
    return Compose([
        ToTensor(),
    ])


class SIEN_dataset(data.Dataset):
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
        super(SIEN_dataset, self).__init__()

        groups = []
        self.group_file = dataset_opt['group_file']
        for i in range(len(self.group_file)):
            groups += [line.rstrip() for line in open(self.group_file[i])]
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = dataset_opt.get('scale', 1)
        self.transform = True
        self.data_augmentation = dataset_opt.get('data_augmentation', None)
        self.patch_size = dataset_opt.get('patch_size', None)

    def __getitem__(self, index):
        
        inputs, target = load_image_train(self.image_filenames[index])
        if self.patch_size!=None:
            inputs, target = udata.get_HWC_patch(inputs, target, self.patch_size, self.upscale_factor)
            
        if self.data_augmentation == 1:
            inputs, target = udata.augment(inputs, target)
        elif self.data_augmentation == 2:
            inputs, target = udata.random_augmentation(inputs, target)

        if self.transform:
            inputs, target = udata.img2tensor([inputs, target], bgr2rgb=True, float32=True)


        return {'lq': inputs, 'gt': target, 'lq_path': self.image_filenames[index][0], 'gt_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)

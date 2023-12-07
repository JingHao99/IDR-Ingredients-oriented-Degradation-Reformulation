import os
import random
import copy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from utils.data_util import crop_HWC_img, random_augmentation, padding, onehot, smooth_one_hot
from sklearn.preprocessing import OneHotEncoder
from data.degradation_util import Degradation


class IDR_dataset(Dataset):
    def __init__(self, dataset_opt):
        super(IDR_dataset, self).__init__()
        self.dataset_opt = dataset_opt
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(dataset_opt)
        self.de_temp = 0
        self.degra_temp = 0
        self.de_type = self.dataset_opt['de_type']
        self.per_degra =self.dataset_opt['per_degra']

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5, 'delowlight': 6, 'clean': 7}
        self.de_label = {'denoise': 0, 'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2, 'deblur': 3, 'delowlight': 4, 'clean': 5}   # degra_type
        self.classes = self.dataset_opt['num_degra']
        self.label_smooth = self.dataset_opt['label_smooth']

        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(dataset_opt['patch_size']),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_blur_ids()
        if 'delowlight' in self.de_type:
            self._init_lowlight_ids()
        if 'clean' in self.de_type:
            self._init_cleanT_ids()

        random.shuffle(self.de_type)

    def _get_group_file(self, file_list):
        groups = []
        for i in range(len(file_list)):
            groups += [line.rstrip() for line in open(file_list[i])]
        ids = [group.split('|') for group in groups]
        return ids

    def _init_clean_ids(self):
        clean_ids = self._get_group_file(self.dataset_opt['denoise_file'])

        if 'denoise_15' in self.de_type:
            self.s15_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = copy.deepcopy(clean_ids)
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)

    def _init_hazy_ids(self):
        self.hazy_ids = self._get_group_file(self.dataset_opt['dehaze_file'])

        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)

    def _init_rs_ids(self):
        self.rs_ids = self._get_group_file(self.dataset_opt['derain_file'])

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)

    def _init_blur_ids(self):
        self.blur_ids = self._get_group_file(self.dataset_opt['deblur_file'])

        self.blur_counter = 0
        self.num_blur = len(self.blur_ids)

    def _init_lowlight_ids(self):
        self.lowlight_ids = self._get_group_file(self.dataset_opt['delowlight_file'])

        self.lowlight_counter = 0
        self.num_lowlight = len(self.lowlight_ids)
    
    def _init_cleanT_ids(self):
        self.cleanT_ids = self._get_group_file(self.dataset_opt['clean_file'])

        self.cleanT_counter = 0
        self.num_cleanT = len(self.cleanT_ids)
    
    def set_per_degra(self,per_degra):
        self.per_degra = per_degra

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.dataset_opt['patch_size'])
        ind_W = random.randint(0, W - self.dataset_opt['patch_size'])

        patch_1 = img_1[ind_H:ind_H + self.dataset_opt['patch_size'], ind_W:ind_W + self.dataset_opt['patch_size']]
        patch_2 = img_2[ind_H:ind_H + self.dataset_opt['patch_size'], ind_W:ind_W + self.dataset_opt['patch_size']]

        return patch_1, patch_2

    def __getitem__(self, _):
        degra_type = self.de_type[self.de_temp]
        de_id = self.de_dict[degra_type]

        de_label = onehot(self.de_label[degra_type],self.classes).float()
        if self.label_smooth:
            de_label = smooth_one_hot(de_label,classes=self.classes,smoothing=self.label_smooth)

        if de_id < 3:
            if de_id == 0:
                clean_id = self.s15_ids[self.s15_counter][1]      # [0] input  [1] target
                self.s15_counter = (self.s15_counter + 1) % self.num_clean
                if self.s15_counter == 0:
                    random.shuffle(self.s15_ids)
            elif de_id == 1:
                clean_id = self.s25_ids[self.s25_counter][1]
                self.s25_counter = (self.s25_counter + 1) % self.num_clean
                if self.s25_counter == 0:
                    random.shuffle(self.s25_ids)
            elif de_id == 2:
                clean_id = self.s50_ids[self.s50_counter][1]
                self.s50_counter = (self.s50_counter + 1) % self.num_clean
                if self.s50_counter == 0:
                    random.shuffle(self.s50_ids)

            clean_img = crop_HWC_img(padding(np.array(Image.open(clean_id).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
            clean_patch= self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            lq_path = clean_id
            clean_patch = random_augmentation(clean_patch)[0]
            degrad_patch = self.D.degrade_single(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                lq_path = self.rs_ids[self.rl_counter][0]
                degrad_img = crop_HWC_img(padding(np.array(Image.open(self.rs_ids[self.rl_counter][0]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
                clean_img = crop_HWC_img(padding(np.array(Image.open(self.rs_ids[self.rl_counter][1]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)

                self.rl_counter = (self.rl_counter + 1) % self.num_rl
                if self.rl_counter == 0:
                    random.shuffle(self.rs_ids)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                lq_path = self.hazy_ids[self.hazy_counter][0]
                degrad_img = crop_HWC_img(padding(np.array(Image.open(self.hazy_ids[self.hazy_counter][0]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
                clean_img = crop_HWC_img(padding(np.array(Image.open(self.hazy_ids[self.hazy_counter][1]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)

                self.hazy_counter = (self.hazy_counter + 1) % self.num_hazy
                if self.hazy_counter == 0:
                    random.shuffle(self.hazy_ids)
            elif de_id == 5:
                # Deblur with GoPro training set
                lq_path = self.blur_ids[self.blur_counter][0]
                degrad_img = crop_HWC_img(padding(np.array(Image.open(self.blur_ids[self.blur_counter][0]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
                clean_img = crop_HWC_img(padding(np.array(Image.open(self.blur_ids[self.blur_counter][1]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)

                self.blur_counter = (self.blur_counter + 1) % self.num_blur
                if self.blur_counter == 0:
                    random.shuffle(self.blur_ids)

            elif de_id == 6:
                # Delowlight with LOL training set
                lq_path = self.lowlight_ids[self.lowlight_counter][0]
                degrad_img = crop_HWC_img(padding(np.array(Image.open(self.lowlight_ids[self.lowlight_counter][0]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
                clean_img = crop_HWC_img(padding(np.array(Image.open(self.lowlight_ids[self.lowlight_counter][1]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)

                self.lowlight_counter = (self.lowlight_counter + 1) % self.num_lowlight
                if self.lowlight_counter == 0:
                    random.shuffle(self.lowlight_ids)

            elif de_id == 7:
                # cleanT with union training set
                lq_path = self.cleanT_ids[self.cleanT_counter][1]
                clean_img = crop_HWC_img(padding(np.array(Image.open(self.cleanT_ids[self.cleanT_counter][1]).convert('RGB')),gt_size=self.dataset_opt['patch_size']), base=16)
                degrad_img = clean_img

                self.cleanT_counter = (self.cleanT_counter + 1) % self.num_cleanT
                if self.cleanT_counter == 0:
                    random.shuffle(self.cleanT_ids)



            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        degrad_patch, clean_patch = self.toTensor(degrad_patch), self.toTensor(clean_patch)

        
        self.degra_temp = ((self.degra_temp + 1) % self.per_degra)
        binary = 1 if self.degra_temp==0 else 0
        self.de_temp = (self.de_temp + binary) % len(self.de_type)
        if self.de_temp == 0 and binary ==1:
            random.shuffle(self.de_type)


        return {'lq_path': lq_path, 'de_id': de_id, 'degrad_patch': degrad_patch, 'clean_patch': clean_patch, 'degra':degra_type, 'de_label':de_label}
       
    def __len__(self):
        return 400 * len(self.dataset_opt['de_type'])
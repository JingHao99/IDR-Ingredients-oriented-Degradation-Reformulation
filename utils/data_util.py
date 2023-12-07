import numpy as np
import cv2
import torch
import random
from torchvision.utils import make_grid
import math


def padding(img, gt_size):
    """
    padding到指定size上
    img_lq (np.float32) 0-1 :
    img_gt (np.float32) 0-1 :
    gt_size (int) :
    cv2.BORDER_REPLICATE/cv2.BORDER_CONSTANT,value=(255,255,255)/cv2.BORDER_REFLECT/cv2.BORDER_REFLECT_101/cv2.BORDER_WRAP"""
    h, w, _ = img.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img

    img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt

# crop an image to the multiple of base
def crop_HWC_img(image, base=64):
    """
    裁切到multiple of base的size上
    :param image: H,W,C
    :param base: (int)
    :return:
    """
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def crop_tensor_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_HWC_patch(input, target, patch_size, scale = 1, ix=-1, iy=-1):
    '''
    :param input: np.float32 (H,W,C)
    :param target: np.float32 (H,W,C)
    :param patch_size:
    :param scale:
    :param ix: start position for height
    :param iy: start position for width
    :return: input & target after crop patch (one patch)
    '''
    ih, iw, channels = input.shape

    tp = scale * patch_size
    ip = tp // scale                    # actual patch size

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    input = input[ix:ix + ip, iy:iy + ip, :]  # [:, ty:ty + tp, tx:tx + tp]
    target = target[ix:ix + ip, iy:iy + ip, :]  # [:, iy:iy + ip, ix:ix + ip]

    return input, target

def get_tensor_patch(input, target, patch_size, scale = 1, ix=-1, iy=-1):
    '''
    :param input: torch.tensor.float32 (B,C,H,W)
    :param target: torch.tensor.float32 (B,C,H,W)
    :param patch_size:
    :param scale:
    :param ix: start position for height
    :param iy: start position for width
    :return: input & target after crop patch (one patch)
    '''
    B,C,ih,iw = input.shape
    # (th, tw) = (scale * ih, scale * iw)

    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    input = input[:, :, ix:ix + ip, iy:iy + ip]  
    target = target[:, :, ix:ix + ip, iy:iy + ip]  

    return input, target

def get_tensor_batch(input, target, mini_batch_size):
    '''
    :param input: torch.tensor.float32 (B,C,H,W)
    :param target: torch.tensor.float32 (B,C,H,W)
    :param batch_size: int
    :return: mini batch
    '''
    B,C,H,W = input.shape
    # (th, tw) = (scale * ih, scale * iw)

    indices = random.sample(range(0, B), k=mini_batch_size)

    input = input[indices]
    target = target[indices]

    return input, target

# if self.data_augmentation:
#     inputs, target = augment(inputs, target, self.hflip, self.rot)
def augment(inputs, target):
    '''
    :param inputs: np.float32 (H,W,C)
    :param target: np.float32 (H,W,C)
    :return: input & target after argumentation
    '''
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot180 = random.random() < 0.5

    def _augment(inputs,target):
        if hflip:
            inputs = inputs[:, ::-1, :]
            target = target[:, ::-1, :]
        if vflip:
            inputs = inputs[::-1, :, :]
            target = target[::-1, :, :]
        if rot180:
            inputs = np.rot90(inputs, k=2)
            target = np.rot90(target, k=2)
        return inputs, target

    inputs, target = _augment(inputs, target)

    return inputs, target


# flip, rotation augmentations
# if self.geometric_augs:
#     img_gt, img_lq = random_augmentation(img_gt, img_lq)
def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def BGR2RGB_toTensor(inputs, target):
    '''
    :param inputs: np.float32 (H,W,C) BGR(imread from cv2)
    :param target: np.float32 (H,W,C)
    :return: torch.tensor(C,H,W) RGB
    '''
    inputs = inputs[:, :, [2, 1, 0]]
    target = target[:, :, [2, 1, 0]]
    inputs = torch.from_numpy(np.ascontiguousarray(np.transpose(inputs, (2, 0, 1)))).float()
    target = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()
    return inputs, target

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, local_rank):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.local_rank = local_rank

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).cuda(self.local_rank)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

def onehot(label: int, classes: int):
    """
    return torch.tensor
    """
    onehot_label = np.zeros([1,classes])
    onehot_label[:,label] = 1
    onehot_label = torch.from_numpy(onehot_label)
    return onehot_label

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))   
    true_dist = torch.empty(size=label_shape)   
    true_dist.fill_(smoothing / (classes - 1))
    _, index = torch.max(true_labels, 1)
    true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)  
    return true_dist


import torch
import numpy as np
from torchvision.utils import make_grid
import utils.util as util
import math
from utils.data_util import get_tensor_patch
from utils.dtype import img_as_ubyte
import torch.nn.functional as F
from PIL import Image
from einops import rearrange


def load_img_tensor(filepath, cuda=True, crop_size=None, pad_size=None):
    '''
    :param filepath:
    :return: (1,C,H,W)  cuda
    '''
    img = np.float32(util.load_img(filepath)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    if pad_size:
        img = pad_tensor(img,pad_size)
    if cuda:
        return img.cuda()
    else:
        return img

def save_img_tensor(restored,result_dir,ippath):
    '''
    :param restored: (1,C,H,W)
    :param result_dir:
    :param ippath:
    :return:
    '''
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    util.save_img(img_as_ubyte(restored),util.Generate_rp(result_dir,ippath))

def save_image_tensor(image_tensor, output_path="output/"):
    image_np = torch_to_np(image_tensor)
    p = np_to_pil(image_np)
    p.save(output_path)

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def pad_tensor(tensor,window_size):
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = tensor.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    tensor = F.pad(tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return tensor

def patchifyBCPHW(imgs,slide_patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, patch_size**2 *3, h, w)
    """
    p = slide_patch_size

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    imgs = rearrange(imgs, 'b c (h p) (w q) -> b (c p q) h w',h=h,w=w,p=p,q=p)
    return imgs

def unpatchifyBCPHW(imgs,slide_patch_size):
    """
    imgs: (N, patch_size**2 *3, h, w)
    return: (N, 3, H, W)
    """
    p = slide_patch_size

    H = imgs.shape[2] * p
    W = imgs.shape[3] * p
    imgs = rearrange(imgs, 'b (c p q) h w -> b c (h p) (w q)',c=3,p=p,q=p)
    return imgs

def patchify(imgs,slide_patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = slide_patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

def unpatchify(x,slide_patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = slide_patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def single_forward(model, inp):
    """PyTorch models forward (single test), it is just a simple warpper
    Args:
        model (PyTorch models)
        inp (Tensor): inputs defined by the models

    Returns:
        output (Tensor): outputs of the models. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch models)
        inp (Tensor): inputs defined by the models

    Returns:
        output (Tensor): outputs of the models. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4
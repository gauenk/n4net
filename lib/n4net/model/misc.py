

# -- linalg --
import numpy as np

# -- data mngmnt --
from easydict import EasyDict as edict

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def get_default_config(sigma):
    cfg = edict()
    cfg.sigma = sigma
    cfg.seed = 123
    cfg.block_w = 64
    cfg.lr = 1e-3
    cfg.epoch_num = 5
    cfg.dset_stride = 1
    cfg.train_batch_size = 4
    cfg.device = "cuda:0"
    return cfg

def calc_padding_rgb(patch_w=5):
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + 13
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + 14 * 2
    offs = total_pad - total_pad0
    return offs,total_pad

def calc_padding(arch_opt):
    patch_w = 5 if arch_opt.rgb else 7
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + 13
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + 14 * 2
    offs = total_pad - total_pad0
    return offs, total_pad

def crop_offset(in_image, row_offs, col_offs):
    if len(row_offs) == 1:
        row_offs += row_offs
    if len(col_offs) == 1:
        col_offs += col_offs
    if row_offs[1] > 0 and col_offs[1] > 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], col_offs[0]:-col_offs[-1]]
    elif row_offs[1] > 0 and col_offs[1] == 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], :]
    elif 0 == row_offs[1] and col_offs[1] > 0:
        out_image = in_image[..., :, col_offs[0]:-col_offs[1]]
    else:
        out_image = in_image
    return out_image

def get_image_params(image, patch_w, neigh_pad):
    im_params = dict()
    im_params['batches'] = image.shape[0]
    im_params['pixels_h'] = image.shape[2] - 2 * neigh_pad
    im_params['pixels_w'] = image.shape[3] - 2 * neigh_pad
    im_params['patches_h'] = im_params['pixels_h'] - (patch_w - 1)
    im_params['patches_w'] = im_params['pixels_w'] - (patch_w - 1)
    im_params['patches'] = im_params['patches_h'] * im_params['patches_w']
    im_params['pad_patches_h'] = image.shape[2] - (patch_w - 1)
    im_params['pad_patches_w'] = image.shape[3] - (patch_w - 1)
    im_params['pad_patches'] = im_params['pad_patches_h'] * im_params['pad_patches_w']
    return im_params

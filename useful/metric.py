'''
    Useful: metrics.
'''

import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def calc_psnr_torch(img1: torch.Tensor, img2: torch.Tensor):
    '''Calc PSNR for torch. img ranges in [0, 1].'''
    return 10. * torch.log10(1. / torch.mean((img1 - img2)**2))


def calc_psnr_ndarray(img1: np.ndarray, img2: np.ndarray):
    '''Calc PSNR for numpy. img ranges in [0, 1] or [0, 255] (dtype=uint8).'''
    return peak_signal_noise_ratio(img1, img2)


def calc_ssim_torch(img1: torch.Tensor, img2: torch.Tensor):
    '''Calc SSIM for torch. img ranges in [0, 1].'''
    return structural_similarity(img1.detach().numpy(), img2.detach().numpy())


def calc_ssim_ndarray(img1: np.ndarray, img2: np.ndarray):
    '''Calc SSIM for numpy. img ranges in [0, 1] or [0, 255] (dtype=uint8).'''
    return structural_similarity(img1, img2)


# Uncomment this if you need LPIPS loss.
# Will load everytime you import this script.
# `pip install lpips` first.
# ############    ############
# from lpips import LPIPS

# lpips = LPIPS(net='vgg')
# lpips.eval()

# def calc_lpips(img1, img2):
#     return lpips(img1, img2).detach().cpu().squeeze().mean()
# ############    ############

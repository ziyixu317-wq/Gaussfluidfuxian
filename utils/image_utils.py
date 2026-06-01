#
# Adapted from 3DGS (Inria, GRAPHDECO research group)
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(3, 1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]
        mse_val = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mse_val = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse_val.float()))
    if mask is not None:
        if torch.isinf(psnr_val).any():
            psnr_val = psnr_val[~torch.isinf(psnr_val)]
    return psnr_val

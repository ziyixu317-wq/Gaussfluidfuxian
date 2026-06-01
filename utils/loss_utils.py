#
# GaussFluids loss functions
# Includes visual losses (L1, SSIM) and physics-based losses
#

import torch
import torch.nn.functional as F
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, height, width) = img1.size()
    real_size = min(window_size, height, width)
    window = create_window(real_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=real_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=real_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=real_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=real_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=real_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ---------------------------------------------------------------------------
# GaussFluids physics-based losses (all on DEFORMED + ACTIVATED state)
# ---------------------------------------------------------------------------

def density_loss(densities):
    """
    Equation (5): L_dens = (1/N) Σ(ρ̄/ρ_i − 1)²
    Args:
        densities: (N,) tensor of SPH density estimates
    """
    rho_bar = densities.mean()
    return ((rho_bar / densities - 1) ** 2).mean()


def volume_loss(scales_activated):
    """
    Equation (7): L_vol = (1/N) Σ(V̄/V_i − 1)²
    Args:
        scales_activated: (N, 3) tensor of exp-activated scales (s_x, s_y, s_z)
    """
    V_i = scales_activated[:, 0] * scales_activated[:, 1] * scales_activated[:, 2]
    V_bar = V_i.mean()
    return ((V_bar / V_i - 1) ** 2).mean()


def anisotropy_loss(scales_activated):
    """
    Equation (8): L_aniso = (1/N) Σ‖s_i / |V_i|^{1/3} − (1,1,1)‖²
    Args:
        scales_activated: (N, 3) tensor of exp-activated scales
    """
    V_i = scales_activated[:, 0] * scales_activated[:, 1] * scales_activated[:, 2]
    det_cbrt = (V_i + 1e-10) ** (1.0 / 3.0)
    isotropic_ref = torch.ones_like(scales_activated)
    return ((scales_activated / det_cbrt.unsqueeze(-1) - isotropic_ref) ** 2).mean()


def opacity_consistency_loss(opacities):
    """
    Equation (9): L_op = MSE(a_i, ā)
    Args:
        opacities: (N, 1) tensor of sigmoid-activated opacities
    """
    a_bar = opacities.mean()
    return ((opacities - a_bar) ** 2).mean()


def light_consistency_loss(sh_coeffs):
    """
    Equation (9): L_light = MSE(c_i, c̄)
    Args:
        sh_coeffs: (N, C) tensor of SH coefficients
    """
    c_bar = sh_coeffs.mean(dim=0, keepdim=True)
    return ((sh_coeffs - c_bar) ** 2).mean()

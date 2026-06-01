#
# GaussFluids render utilities
#

import torch


def get_state_at_time(gaussians, time):
    """
    Query the spatio-temporal encoder to get the deformed particle state
    at a given time.

    Args:
        gaussians: GaussianFluidParticles model
        time: scalar float or tensor

    Returns:
        (points, scales_final, rotations_final, opacity_final, shs_final)
        All in activated (ready-to-render) space
    """
    N = gaussians._xyz.shape[0]
    time_tensor = torch.tensor([time], dtype=torch.float32, device=gaussians._xyz.device)

    # Get deformed state from spatio-temporal encoder
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
        gaussians.get_deformed_state(time_tensor)

    # Activate
    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)

    return means3D_final, scales_final, rotations_final, opacity_final, shs_final

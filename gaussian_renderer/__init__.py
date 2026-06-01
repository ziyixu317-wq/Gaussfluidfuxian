#
# GaussFluids: Gaussian Splatting Renderer
# Adapted from 3DGS for spatio-temporal deformation
#

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer
)


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, is_training=True):
    """
    Render GaussFluids particles via splatting.

    Applies spatio-temporal deformation before rasterization.

    Args:
        viewpoint_camera: Camera with .FoVx, .FoVy, .image_height, .image_width,
                          .world_view_transform, .full_proj_transform, .camera_center, .time
        pc: GaussianFluidParticles model
        pipe: PipelineParams
        bg_color: background color tensor on GPU
        scaling_modifier: global scale modifier
        override_color: optional color override
        is_training: whether in training mode (affects gradient tracking)

    Returns:
        dict with: render, viewspace_points, visibility_filter, radii, depth
    """
    # Setup screen-space points for gradient tracking
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Setup rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get base attributes
    means3D = pc.get_xyz
    opacity = pc._opacity

    # Query spatio-temporal encoder for deformed state
    time = viewpoint_camera.time
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
        pc.get_deformed_state(time)

    # Activate for rendering
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # Color preprocessing
    colors_precomp = None
    cov3D_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            from utils.sh_utils import eval_sh
            shs_view = shs_final.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = (means3D_final - viewpoint_camera.camera_center.repeat(
                shs_final.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    # Rasterize
    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=screenspace_points,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    # For training: return deformed state for physics loss computation
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "deformed_xyz": means3D_final,
            "scales_activated": scales_final,
            "opacities_activated": opacity,
            "shs": shs_final,
        }
    else:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
        }

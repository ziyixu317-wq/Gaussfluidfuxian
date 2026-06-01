#
# GaussFluids: Custom PLY Deformation
# Loads a manually-edited PLY file and propagates changes through time
# using the trained spatio-temporal encoder.
# Analogous to deform_custom_ply.py in the reference codebase.
#

import os
import sys
import argparse
import numpy as np
import torch
import imageio
from plyfile import PlyData
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, GaussFluidsParams
from scene import Scene
from scene.gaussian_fluid_particles import GaussianFluidParticles
from scene.camera import MiniCam
from gaussian_renderer import render
from utils.general_utils import safe_state, inverse_sigmoid
from utils.render_utils import get_state_at_time
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def load_custom_ply(ply_path, gaussians, canon_xyz, canon_scaling,
                    canon_rotation, canon_opacity, canon_shs, canon_transform_feature):
    """
    Load a user-edited PLY file and map it to GaussianFluidParticles.

    If the PLY has 'original_index' column, uses index-based matching.
    Otherwise, reads PLY values directly.

    Returns True if successful.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Check for original_index
    has_orig_idx = 'original_index' in vertex.data.dtype.names

    if has_orig_idx:
        print("Found original_index column, using index-based mapping...")
        orig_indices = np.asarray(vertex['original_index'])

        edited_xyz = np.stack([
            np.asarray(vertex['x']),
            np.asarray(vertex['y']),
            np.asarray(vertex['z'])
        ], axis=1)

        # Map back to canonical space via index
        new_xyz = torch.tensor(edited_xyz, dtype=torch.float32, device="cuda")
        new_scaling = torch.tensor(canon_scaling[orig_indices], dtype=torch.float32,
                                   device="cuda")
        new_rotation = torch.tensor(canon_rotation[orig_indices], dtype=torch.float32,
                                    device="cuda")
        new_opacity = torch.tensor(canon_opacity[orig_indices], dtype=torch.float32,
                                   device="cuda")
        new_shs = torch.tensor(canon_shs[orig_indices], dtype=torch.float32, device="cuda")
        new_tf = torch.tensor(canon_transform_feature[orig_indices], dtype=torch.float32,
                              device="cuda")
    else:
        print("No original_index, reading PLY values directly...")
        xyz = np.stack([
            np.asarray(vertex['x']), np.asarray(vertex['y']), np.asarray(vertex['z'])
        ], axis=1)

        f_dc = np.stack([
            np.asarray(vertex['f_dc_0']),
            np.asarray(vertex['f_dc_1']),
            np.asarray(vertex['f_dc_2'])
        ], axis=1)

        num_sh = (gaussians.max_sh_degree + 1) ** 2
        f_rest = np.zeros((xyz.shape[0], (num_sh - 1) * 3))
        for i in range((num_sh - 1) * 3):
            col_name = f'f_rest_{i}'
            if col_name in vertex.data.dtype.names:
                f_rest[:, i] = np.asarray(vertex[col_name])

        opacity = np.asarray(vertex['opacity'])

        scale_names = sorted(
            [n for n in vertex.data.dtype.names if n.startswith('scale_')],
            key=lambda x: int(x.split('_')[-1])
        )
        scales = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=1)

        rot_names = sorted(
            [n for n in vertex.data.dtype.names if n.startswith('rot_')],
            key=lambda x: int(x.split('_')[-1])
        )
        rots = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=1)

        new_xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        new_scaling = torch.tensor(scales, dtype=torch.float32, device="cuda")
        new_rotation = torch.tensor(rots, dtype=torch.float32, device="cuda")
        new_opacity = torch.tensor(opacity, dtype=torch.float32, device="cuda").unsqueeze(-1)

        # Combine DC + rest SH
        shs_full = np.zeros((xyz.shape[0], num_sh, 3))
        shs_full[:, 0, :] = f_dc
        for i in range(num_sh - 1):
            shs_full[:, i + 1, :] = f_rest[:, i * 3:(i + 1) * 3]
        new_shs = torch.tensor(shs_full, dtype=torch.float32, device="cuda")

        # Transform features: use canonical ones
        new_tf = torch.tensor(canon_transform_feature[:xyz.shape[0]],
                              dtype=torch.float32, device="cuda")

    # Apply to Gaussians
    gaussians._xyz = torch.nn.Parameter(new_xyz)
    gaussians._scaling = torch.nn.Parameter(new_scaling)
    gaussians._rotation = torch.nn.Parameter(new_rotation)
    gaussians._opacity = torch.nn.Parameter(new_opacity)

    # Reshape SH: (N, num_sh, 3) → standard format (N, num_sh, 3)
    # features_dc: (N, 1, 3), features_rest: (N, num_sh-1, 3)
    shs_tensor = new_shs
    gaussians._features_dc = torch.nn.Parameter(
        shs_tensor[:, 0:1, :].contiguous()
    )
    gaussians._features_rest = torch.nn.Parameter(
        shs_tensor[:, 1:, :].contiguous()
    )
    gaussians._transform_feature = torch.nn.Parameter(new_tf)

    gaussians.max_radii2D = torch.zeros((gaussians._xyz.shape[0]), device="cuda")

    return True


def deform_custom_ply(args, gaussfluids_params):
    """
    Load trained GaussFluids model and user-edited PLY,
    then deform through time and render/save.
    """
    # Load original model
    gaussians = GaussianFluidParticles(
        sh_degree=args.sh_degree,
        gaussfluids_params=vars(gaussfluids_params)
    )
    scene = Scene(args, gaussians, load_iteration=args.load_iteration)
    gaussians.spatio_temporal_encoder.cuda()

    # Save canonical state for reference
    canon_xyz = gaussians._xyz.detach().cpu().numpy().copy()
    canon_scaling = gaussians._scaling.detach().cpu().numpy().copy()
    canon_rotation = gaussians._rotation.detach().cpu().numpy().copy()
    canon_opacity = gaussians._opacity.detach().cpu().numpy().copy()
    canon_shs = gaussians.get_features.detach().cpu().numpy().copy()
    canon_tf = gaussians._transform_feature.detach().cpu().numpy().copy()

    # Load custom PLY
    ply_path = args.custom_ply
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Custom PLY not found: {ply_path}")

    success = load_custom_ply(
        ply_path, gaussians, canon_xyz, canon_scaling,
        canon_rotation, canon_opacity, canon_shs, canon_tf
    )
    if not success:
        print("Failed to load custom PLY!")
        return

    print(f"Loaded custom PLY with {gaussians._xyz.shape[0]} particles")

    # Get unique timestamps
    train_cameras = scene.getTrainCameras()
    unique_times = sorted(set(cam.time for cam in train_cameras))

    # Output directory
    output_path = os.path.join(args.model_path, "custom_deform_output")
    os.makedirs(output_path, exist_ok=True)

    # Setup fixed camera for video rendering
    bg_color = scene.background

    # Use first camera view as fixed viewpoint
    ref_cam = train_cameras[0]

    rendered_frames = []
    save_ply = getattr(args, 'save_ply', False)

    print(f"Deforming through {len(unique_times)} timesteps...")
    for t in tqdm(unique_times, desc="Deforming"):
        # Get deformed state at time t
        points, scales_final, rotations_final, opacity_final, shs_final = \
            get_state_at_time(gaussians, t)

        if save_ply:
            from export_perframe_gaussfluids import init_3DGaussians_ply
            gs_ply = init_3DGaussians_ply(
                points=points,
                scales=scales_final,
                rotations=rotations_final,
                opacity=opacity_final,
                shs=shs_final,
                transform_features=gaussians._transform_feature,
                sh_degree=args.sh_degree,
                original_indices=torch.arange(points.shape[0], device=points.device),
            )
            gs_ply.write(os.path.join(output_path, f"deformed_t{t:.3f}.ply"))

        # Render from fixed camera with current time
        mini_cam = MiniCam(
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            fovy=ref_cam.FoVy,
            fovx=ref_cam.FoVx,
            znear=ref_cam.znear,
            zfar=ref_cam.zfar,
            world_view_transform=ref_cam.world_view_transform,
            full_proj_transform=ref_cam.full_proj_transform,
            time=t,
        )

        render_pkg = render(mini_cam, gaussians, args_pipe, bg_color, is_training=False)
        rendered_image = render_pkg["render"]

        # Convert to numpy for video
        frame = torch.clamp(rendered_image, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        rendered_frames.append(frame)

    # Save video
    video_path = os.path.join(output_path, "custom_deform_video.mp4")
    imageio.mimsave(video_path, rendered_frames, fps=30)
    print(f"Saved video to {video_path}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaussFluids Custom PLY Deformation")
    lp = ModelParams()
    pp = PipelineParams()
    gp = GaussFluidsParams()

    lp.add_arguments(parser)
    pp.add_arguments(parser)
    gp.add_arguments(parser)

    parser.add_argument('--load_iteration', type=int, required=True,
                      help='Checkpoint iteration to load')
    parser.add_argument('--custom_ply', type=str, required=True,
                      help='Path to manually-edited PLY file')
    parser.add_argument('--save_ply', action='store_true', default=False,
                      help='Save deformed PLY at each timestep')

    args = parser.parse_args(sys.argv[1:])

    # Assign args
    for attr in ['sh_degree', 'source_path', 'model_path', 'white_background',
                 'resolution', 'eval', 'load_iteration']:
        if hasattr(args, attr):
            setattr(lp, attr, getattr(args, attr))

    for attr in ['transform_feature_dim', 'mlp_hidden_dim', 'mlp_num_layers',
                 'smoothing_length', 'knn_k', 'time_pe_freqs', 'physics_loss_stride']:
        if hasattr(args, attr):
            setattr(gp, attr, getattr(args, attr))

    # Pipeline params
    for attr in ['convert_SHs_python', 'compute_cov3D_python', 'debug']:
        if hasattr(args, attr):
            setattr(pp, attr, getattr(args, attr))

    args_pipe = pp

    safe_state(False)
    deform_custom_ply(lp, gp)
    print("All done!")

#
# GaussFluids: Per-Frame Gaussian Export
# Exports deformed particle state as 3DGS PLY files at each test timestep.
# Analogous to new_export_perframe_3DGS.py in the reference codebase.
#

import os
import sys
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, GaussFluidsParams
from scene import Scene
from scene.gaussian_fluid_particles import GaussianFluidParticles
from utils.general_utils import safe_state
from utils.render_utils import get_state_at_time


def init_3DGaussians_ply(points, scales, rotations, opacity, shs, transform_features,
                         sh_degree=1, original_indices=None):
    """
    Create a PlyData object with full 3DGS attributes + GaussFluids-specific fields.

    Args:
        points: (N, 3) deformed positions
        scales: (N, 3) activated scales
        rotations: (N, 4) normalized quaternions
        opacity: (N, 1) sigmoid-activated opacities
        shs: (N, (deg+1)^2, 3) SH coefficients (logit-space, not activated)
        transform_features: (N, D) per-particle transform features
        sh_degree: SH degree (1 for GaussFluids)
        original_indices: (N,) original particle indices
    """
    N = points.shape[0]
    num_sh = (sh_degree + 1) ** 2  # 4 for degree 1

    # Reshape SH: (N, 3, num_sh) → (N, num_sh*3)
    shs_flat = shs.reshape(N, -1).detach().cpu().numpy()

    # Separate DC (first 3 values = 1 per channel) and rest
    f_dc = shs_flat[:, :3]   # (N, 3)
    f_rest = shs_flat[:, 3:num_sh*3]   # (N, rest)

    # Build attribute list
    xyz = points.detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    # Convert opacity to logit-space for PLY storage
    opacity_logit = torch.log(opacity / (1.0 - opacity + 1e-10) + 1e-10)
    opacity_arr = opacity_logit.detach().cpu().numpy()

    scales_log = torch.log(scales + 1e-10).detach().cpu().numpy()
    rotations_arr = rotations.detach().cpu().numpy()
    transform_features_arr = transform_features.detach().cpu().numpy()

    # Build PLY dtype
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ]
    for i in range(3):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    num_rest = max(0, num_sh * 3 - 3)
    for i in range(num_rest):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    dtype_full.append(('opacity', 'f4'))
    for i in range(3):
        dtype_full.append((f'scale_{i}', 'f4'))
    for i in range(4):
        dtype_full.append((f'rot_{i}', 'f4'))
    if original_indices is not None:
        dtype_full.append(('original_index', 'i4'))

    attributes = np.concatenate(
        [xyz, normals, f_dc, f_rest, opacity_arr, scales_log, rotations_arr],
        axis=1
    )
    if original_indices is not None:
        attributes = np.concatenate(
            [attributes, original_indices.detach().cpu().numpy().reshape(-1, 1)], axis=1
        )

    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    return PlyData([PlyElement.describe(elements, 'vertex')])


def export_perframe(args, gaussfluids_params):
    """
    Export per-frame PLY files for all test cameras.
    """
    gaussians = GaussianFluidParticles(
        sh_degree=args.sh_degree,
        gaussfluids_params=vars(gaussfluids_params)
    )
    scene = Scene(args, gaussians)
    gaussians.spatio_temporal_encoder.cuda()

    # Output directory
    output_path = os.path.join(args.model_path, "perframe_export")
    os.makedirs(output_path, exist_ok=True)

    # Collect unique timestamps
    test_cameras = scene.getTestCameras()
    time_to_cameras = {}
    for cam in test_cameras:
        t = cam.time
        if t not in time_to_cameras:
            time_to_cameras[t] = []
        time_to_cameras[t].append(cam)

    unique_times = sorted(time_to_cameras.keys())
    print(f"Exporting {len(unique_times)} unique timestamps...")

    for frame_idx, t in enumerate(tqdm(unique_times, desc="Exporting frames")):
        # Get deformed state at time t
        points, scales_final, rotations_final, opacity_final, shs_final = \
            get_state_at_time(gaussians, t)

        # Create PLY
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

        ply_path = os.path.join(output_path, f"time_{frame_idx:05d}.ply")
        gs_ply.write(ply_path)

    # Also export canonical (t=0) state
    points, scales_final, rotations_final, opacity_final, shs_final = \
        get_state_at_time(gaussians, 0.0)
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
    gs_ply.write(os.path.join(output_path, "canonical.ply"))

    print(f"Exported {len(unique_times) + 1} PLY files to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaussFluids Per-Frame Export")
    lp = ModelParams()
    pp = PipelineParams()
    gp = GaussFluidsParams()

    lp.add_arguments(parser)
    pp.add_arguments(parser)
    gp.add_arguments(parser)

    parser.add_argument('--load_iteration', type=int, required=True,
                      help='Checkpoint iteration to load')

    args = parser.parse_args(sys.argv[1:])

    # Assign args
    for attr in ['sh_degree', 'source_path', 'model_path', 'white_background',
                 'resolution', 'eval', 'load_iteration']:
        if hasattr(args, attr):
            setattr(args, attr, getattr(args, attr))

    for attr in ['transform_feature_dim', 'mlp_hidden_dim', 'mlp_num_layers',
                 'smoothing_length', 'knn_k', 'time_pe_freqs', 'physics_loss_stride']:
        if hasattr(args, attr):
            setattr(gp, attr, getattr(args, attr))

    safe_state(False)
    export_perframe(args, gp)
    print("Done!")

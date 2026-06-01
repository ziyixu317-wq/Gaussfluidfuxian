#
# GaussFluids: Gaussian Fluid Particles Model
# Paper Section 3.1, 3.2, 3.3
#
# Extends 3DGS with:
#  - Per-particle transform features q (dim=64)
#  - Spatio-Temporal Encoder (shared MLP F)
#  - SPH density estimation (KNN, Poly6 kernel)
#  - Physics-aware densification
#

import torch
import torch.nn as nn
import numpy as np
import os
from plyfile import PlyData, PlyElement

from utils.general_utils import inverse_sigmoid, build_rotation, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud, batch_quaternion_multiply
from utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from scene.spatio_temporal_encoding import SpatioTemporalEncoder
from scene.density_optimization import compute_sph_density


class GaussianFluidParticles(nn.Module):
    """
    GaussFluids core model representing fluid particles as 3D Gaussian primitives
    with per-particle spatio-temporal transform features.
    """

    def __init__(self, sh_degree: int = 1, gaussfluids_params=None):
        """
        Args:
            sh_degree: spherical harmonics degree (1 for GaussFluids per paper)
            gaussfluids_params: configuration dict with transform_feature_dim, etc.
        """
        super().__init__()

        # SH degree: GaussFluids uses 1st order only (paper Sec 3.1)
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree

        # GaussFluids hyperparams
        if gaussfluids_params is None:
            gaussfluids_params = {}
        self.transform_feature_dim = gaussfluids_params.get('transform_feature_dim', 64)
        self.mlp_hidden_dim = gaussfluids_params.get('mlp_hidden_dim', 256)
        self.mlp_num_layers = gaussfluids_params.get('mlp_num_layers', 4)
        self.time_pe_freqs = gaussfluids_params.get('time_pe_freqs', 10)
        self.smoothing_length = gaussfluids_params.get('smoothing_length', 0.3)
        self.knn_k = gaussfluids_params.get('knn_k', 64)
        self.physics_loss_stride = gaussfluids_params.get('physics_loss_stride', 5)

        # Gaussian parameters (created when initialized from point cloud)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._transform_feature = torch.empty(0)  # per-particle feature q

        # Spatio-temporal encoder (shared MLP F)
        self.spatio_temporal_encoder = SpatioTemporalEncoder(
            feature_dim=self.transform_feature_dim,
            hidden_dim=self.mlp_hidden_dim,
            num_layers=self.mlp_num_layers,
            time_pe_freqs=self.time_pe_freqs,
        )

        # Activation functions
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1)

        # Adaptive control tracking
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_transform_feature(self):
        return self._transform_feature

    def get_covariance(self, scaling_modifier=1):
        return build_scaling_rotation(
            scaling_modifier * self.scaling_activation(self._scaling),
            self._rotation
        )

    def oneupSHdegree(self):
        pass  # GaussFluids only uses degree 1

    # ------------------------------------------------------------------
    # Initialization from point cloud
    # ------------------------------------------------------------------

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussian particles from a point cloud.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # SH: degree 1 → (1+1)^2 = 4 coefficients per channel
        # features_dc: (N, 1, 3) for 0th order
        # features_rest: (N, 3, 3) for 1st order (4-1=3 rest coeffs per channel)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               device="cuda").float()
        features[:, :3, 0] = fused_color  # DC component
        features[:, :3, 1:] = 0.0  # Higher order (1st) initialized to 0

        # Scale initialization: avg distance to 3 nearest neighbors
        # Uses our multi-backend KNN (pytorch3d/faiss/torch.cdist)
        from scene.density_optimization import knn_avg_neighbor_dist
        avg_dist = knn_avg_neighbor_dist(fused_point_cloud, k=4)

        scales = torch.log(avg_dist.clamp(min=1e-7)).unsqueeze(-1).repeat(1, 3)

        # Random rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1  # identity quaternion

        # Opacity initialized to 0.1 (inverse sigmoid)
        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1),
                                                      dtype=torch.float, device="cuda"))

        # Transform features initialized to zero (identity mapping at t=0)
        transform_features = torch.zeros(
            (fused_point_cloud.shape[0], self.transform_feature_dim),
            device="cuda"
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous()
                                         .requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous()
                                          .requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._transform_feature = nn.Parameter(transform_features.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    # ------------------------------------------------------------------
    # Spatio-Temporal Deformation (Equation 4)
    # ------------------------------------------------------------------

    def get_deformed_state(self, time):
        """
        Compute deformed particle state at given time t.
        Equation (4): ΔA_t = F(q, t), A_t = A_0 + ΔA_t

        Args:
            time: scalar or (N, 1) tensor of time values

        Returns:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final
            Note: scales are still in log-space, rotations are raw quaternion,
                  opacity is in logit-space — caller must activate.
        """
        delta_p, delta_s, delta_r = self.spatio_temporal_encoder(
            self._transform_feature, time
        )

        # Position: additive offset
        means3D_final = self._xyz + delta_p

        # Scale: additive offset in log-space
        scales_final = self._scaling + delta_s

        # Rotation: output_head bias = [1,0,0,0] → zero-weight MLP outputs
        # identity quaternion. DO NOT pre-normalize — batch_quaternion_multiply
        # and renderer's rotation_activation handle it. Pre-normalizing kills
        # gradient flow through the quaternion.
        rotations_final = batch_quaternion_multiply(delta_r, self._rotation)

        # Opacity: no deformation (paper only deforms p, s, r)
        opacity_final = self._opacity

        # SH features: no time-dependent deformation
        shs_final = self.get_features

        return means3D_final, scales_final, rotations_final, opacity_final, shs_final

    # ------------------------------------------------------------------
    # Physics losses (computed on DEFORMED + ACTIVATED state)
    # ------------------------------------------------------------------

    def compute_physics_losses(self, p_t, scales_activated, opacities_activated,
                               shs, iteration, lambda_weights):
        """
        Compute physics-based loss terms.

        Args:
            p_t: (N, 3) deformed positions
            scales_activated: (N, 3) exp-activated scales
            opacities_activated: (N, 1) sigmoid-activated opacities
            shs: (N, C) SH coefficients
            iteration: current training iteration
            lambda_weights: dict of current loss weights

        Returns:
            dict of loss tensors (scalar values)
        """
        from utils.loss_utils import (
            density_loss, volume_loss, anisotropy_loss,
            opacity_consistency_loss, light_consistency_loss
        )

        losses = {}

        # SPH density via KNN (expensive, skip when λ_d=0 or not on stride)
        dens_w = lambda_weights.get('lambda_dens', 0)
        if dens_w > 0:
            stride = self.physics_loss_stride
            if iteration % stride == 0:
                densities = compute_sph_density(p_t, h=self.smoothing_length, k=self.knn_k)
                losses['L_dens'] = dens_w * density_loss(densities)

        if lambda_weights.get('lambda_vol', 0) > 0:
            losses['L_vol'] = lambda_weights['lambda_vol'] * volume_loss(scales_activated)

        if lambda_weights.get('lambda_aniso', 0) > 0:
            losses['L_aniso'] = lambda_weights['lambda_aniso'] * anisotropy_loss(scales_activated)

        if lambda_weights.get('lambda_op', 0) > 0:
            losses['L_op'] = lambda_weights['lambda_op'] * \
                opacity_consistency_loss(opacities_activated)

        if lambda_weights.get('lambda_light', 0) > 0:
            losses['L_light'] = lambda_weights['lambda_light'] * \
                light_consistency_loss(shs)

        return losses

    # ------------------------------------------------------------------
    # Adaptive Density Control (Densification)
    # θ handling: clone=copy, split=inherit
    # ------------------------------------------------------------------

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone Gaussians with high gradient and small scale."""
        # Extract points for cloning
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling_activation(self._scaling), dim=1).values <=
            self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        # Clone transform features: exact copy
        new_transform_feature = self._transform_feature[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_scaling, new_rotation, new_opacity,
            new_transform_feature
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradient."""
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling_activation(self._scaling), dim=1).values >
            self.percent_dense * scene_extent
        )

        stds = self.scaling_activation(self._scaling[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(self.scaling_activation(
            self._scaling[selected_pts_mask]) / (0.8 * N)
        ).repeat(N, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        # Split: inherit transform feature from parent (no noise added)
        new_transform_feature = self._transform_feature[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_scaling, new_rotation, new_opacity,
            new_transform_feature
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(),
                                            device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        """Prune Gaussian points. Also prunes transform features."""
        valid_points_mask = ~mask
        pruned_optimizer_state = self._optimizer_state_dict_with_transform_feature(
            valid_points_mask, self.optimizer.state_dict()
        )

        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._transform_feature = nn.Parameter(
            self._transform_feature[valid_points_mask]
        )
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        # Update optimizer state
        self.optimizer.load_state_dict(pruned_optimizer_state)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_scaling, new_rotation, new_opacity,
                              new_transform_feature):
        """Add new Gaussians to the model and update optimizer."""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "transform_feature": new_transform_feature,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._transform_feature = optimizable_tensors["transform_feature"]

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Concatenate tensors and register them in the optimizer.
        Only touches groups whose names are in tensors_dict.
        Skips groups like 'mlp' (encoder weights) that don't change during densification.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in tensors_dict:
                continue  # skip encoder/mlp params — they don't grow

            stored_state = self.optimizer.state.get(
                group['params'][0], None
            )
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(tensors_dict[name])),
                    dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(tensors_dict[name])),
                    dim=0
                )
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], tensors_dict[name]), dim=0
                    )
                )
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], tensors_dict[name]), dim=0
                    )
                )
                optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def _optimizer_state_dict_with_transform_feature(self, mask, optimizer_state_dict):
        """Prune optimizer state to match pruned tensors.
        Only prunes per-particle groups; skips encoder/mlp params (fixed size).
        """
        param_groups = optimizer_state_dict["param_groups"]
        param_state = optimizer_state_dict["state"]
        mask_len = mask.shape[0]

        for group in param_groups:
            stored_state = param_state.get(id(group["params"][0]), None)
            if stored_state is not None:
                for key in ["exp_avg", "exp_avg_sq"]:
                    if key in stored_state and stored_state[key].shape[0] == mask_len:
                        stored_state[key] = stored_state[key][mask]
                del param_state[id(group["params"][0])]

        return optimizer_state_dict

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate 2D position gradients for densification decisions."""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def reset_opacity(self):
        """Reset opacity of all Gaussians to a lower value."""
        opacities_new = inverse_sigmoid(
            torch.min(self.opacity_activation(self._opacity), torch.ones_like(self._opacity) * 0.05)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        """Replace a tensor in the optimizer parameter groups."""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        """Set up optimizer with separate param groups including transform feature."""
        self.percent_dense = training_args.percent_dense if hasattr(training_args, 'percent_dense') else 0.01

        l = [
            {'params': [self._xyz],
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "xyz"},
            {'params': [self._features_dc],
             'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest],
             'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity],
             'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling],
             'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation],
             'lr': training_args.rotation_lr, "name": "rotation"},
            # GaussFluids: transform feature + spatio-temporal encoder
            {'params': [self._transform_feature],
             'lr': training_args.transform_feature_lr, "name": "transform_feature"},
            {'params': list(self.spatio_temporal_encoder.parameters()),
             'lr': training_args.mlp_lr, "name": "mlp"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    # ------------------------------------------------------------------
    # Freeze / unfreeze MLP for phase transitions
    # ------------------------------------------------------------------

    def freeze_mlp(self):
        """Freeze the spatio-temporal encoder (Phase 1)."""
        for param in self.spatio_temporal_encoder.parameters():
            param.requires_grad = False

    def unfreeze_mlp(self):
        """Unfreeze the spatio-temporal encoder (Phase 2+)."""
        for param in self.spatio_temporal_encoder.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_ply(self, path):
        """Save Gaussian particles as PLY file."""
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        transform_feature = self._transform_feature.detach().cpu().numpy()

        # Build PLY attributes
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                      ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        # DC SH: 3 channels
        for i in range(3):
            dtype_full.append(('f_dc_{}'.format(i), 'f4'))
        # Rest SH: (degree+1)^2 - 1 = 3 for degree 1
        num_rest = (self.max_sh_degree + 1) ** 2 - 1
        for i in range(num_rest * 3):
            dtype_full.append(('f_rest_{}'.format(i), 'f4'))
        dtype_full.append(('opacity', 'f4'))
        for i in range(3):
            dtype_full.append(('scale_{}'.format(i), 'f4'))
        for i in range(4):
            dtype_full.append(('rot_{}'.format(i), 'f4'))
        for i in range(self.transform_feature_dim):
            dtype_full.append(('transform_feature_{}'.format(i), 'f4'))

        attributes = np.concatenate(
            [xyz, normals, f_dc, f_rest, opacities, scale, rotation, transform_feature],
            axis=1
        )
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        """Load Gaussian particles from PLY file."""
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        num_rest = (self.max_sh_degree + 1) ** 2 - 1
        features_rest = np.zeros((xyz.shape[0], num_rest, 3))
        for i in range(num_rest * 3):
            features_rest[:, i // 3, i % 3] = np.asarray(
                plydata.elements[0]["f_rest_{}".format(i)]
            )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load transform features if present
        tf_names = [p.name for p in plydata.elements[0].properties
                    if p.name.startswith("transform_feature_")]
        if len(tf_names) > 0:
            tf_names = sorted(tf_names, key=lambda x: int(x.split('_')[-1]))
            transform_features = np.zeros((xyz.shape[0], len(tf_names)))
            for idx, attr_name in enumerate(tf_names):
                transform_features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            transform_features = np.zeros((xyz.shape[0], self.transform_feature_dim))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda")
                                 .requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda")
                                        .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_rest, dtype=torch.float, device="cuda")
                                          .transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda")
                                     .requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda")
                                     .requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda")
                                     .requires_grad_(True))
        self._transform_feature = nn.Parameter(
            torch.tensor(transform_features, dtype=torch.float, device="cuda")
            .requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

    def save_deformation(self, path):
        """Save spatio-temporal encoder weights."""
        torch.save(self.spatio_temporal_encoder.state_dict(),
                   os.path.join(path, "spatio_temporal_encoder.pth"))

    def load_deformation(self, path):
        """Load spatio-temporal encoder weights."""
        self.spatio_temporal_encoder.load_state_dict(
            torch.load(os.path.join(path, "spatio_temporal_encoder.pth"))
        )

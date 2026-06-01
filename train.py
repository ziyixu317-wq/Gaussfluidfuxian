#
# GaussFluids: Main Training Script
# 3-phase training with dynamic weight scheduling
# Paper Sections 3.1-3.3, 4.1.2
#

import os
import sys
import math
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from arguments import ModelParams, PipelineParams, GaussFluidsParams, OptimizationParams
from scene import Scene
from scene.gaussian_fluid_particles import GaussianFluidParticles
from gaussian_renderer import render
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.render_utils import get_state_at_time
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dynamic weight scheduling (3 phases per paper Section 4.1.2)
# ---------------------------------------------------------------------------

def get_dynamic_lambda_weights(iteration, args_training, phase="phase1"):
    """
    Compute dynamic loss weights based on training phase.

    Per paper Section 4.1.2:
    - Phase 1: lambda_d=0, lambda_a=10%, lambda_v=10%
                ("while lambda_a and lambda_v are set to 10% of their final values")
    - Phase 2: lambda_d linearly increases from 0 to 10% of final
    - Phase 3: lambda_a, lambda_v, lambda_d gradually raised to final values
    - lambda_light (lambda_l) is NEVER dynamically scheduled — paper does not mention it

    Returns:
        dict of weight values to apply
    """
    # Static weights (never dynamically scheduled by paper)
    weights = {
        'lambda_rgb': args_training.lambda_rgb,
        'lambda_ssim': args_training.lambda_ssim,
        'lambda_op': args_training.lambda_op,
        'lambda_light': args_training.lambda_light,  # always full value per paper
    }

    if phase == "phase1":
        # Phase 1: lambda_d=0, lambda_a=10%, lambda_v=10% (paper explicitly states)
        weights['lambda_dens'] = 0.0
        weights['lambda_aniso'] = args_training.lambda_aniso * 0.1
        weights['lambda_vol'] = args_training.lambda_vol * 0.1
    elif phase == "phase2":
        # Phase 2: lambda_dens linearly increases from 0 to 10% of final
        # lambda_a and lambda_v stay at 10% (not yet ramping)
        phase2_total = args_training.phase3_start - args_training.phase2_start
        phase2_progress = (iteration - args_training.phase2_start) / max(phase2_total, 1)
        dens_ratio = min(phase2_progress, 1.0) * 0.1
        weights['lambda_dens'] = args_training.lambda_dens * dens_ratio
        weights['lambda_aniso'] = args_training.lambda_aniso * 0.1
        weights['lambda_vol'] = args_training.lambda_vol * 0.1
    elif phase == "phase3":
        # Phase 3: lambda_a, lambda_v, lambda_d all gradually raised to final
        phase3_total = args_training.iterations - args_training.phase3_start
        phase3_progress = (iteration - args_training.phase3_start) / max(phase3_total, 1)
        phase3_progress = min(phase3_progress, 1.0)

        # lambda_dens: from 10% to 100%
        dens_ratio = 0.1 + 0.9 * phase3_progress
        weights['lambda_dens'] = args_training.lambda_dens * dens_ratio
        # lambda_aniso: from 10% to 100%
        aniso_ratio = 0.1 + 0.9 * phase3_progress
        weights['lambda_aniso'] = args_training.lambda_aniso * aniso_ratio
        # lambda_vol: from 10% to 100%
        vol_ratio = 0.1 + 0.9 * phase3_progress
        weights['lambda_vol'] = args_training.lambda_vol * vol_ratio
    else:
        weights['lambda_dens'] = args_training.lambda_dens
        weights['lambda_aniso'] = args_training.lambda_aniso
        weights['lambda_vol'] = args_training.lambda_vol

    return weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, gaussfluids_params, testing_iterations,
             saving_iterations, checkpoint_iterations, debug_from):
    """
    GaussFluids 3-phase training loop.
    """
    first_iter = 0
    tb_writer = SummaryWriter(os.path.join(dataset.model_path, "logs"))

    # Initialize model (encoder and particle params created on CPU initially)
    gaussians = GaussianFluidParticles(
        sh_degree=dataset.sh_degree,
        gaussfluids_params=vars(gaussfluids_params)
    )

    # Initialize scene (calls create_from_pcd: particle params now on CUDA)
    scene = Scene(dataset, gaussians)

    # Load checkpoint if specified
    if dataset.load_iteration is not None:
        first_iter = dataset.load_iteration

    # Move encoder to CUDA BEFORE optimizer creation so param refs are correct
    gaussians.spatio_temporal_encoder.cuda()

    # Register all (now GPU-resident) params with optimizer
    gaussians.training_setup(opt)

    # Setup iterators
    train_cameras = scene.getTrainCameras()
    t0_cameras = scene.getT0Cameras()

    # Learning rate schedulers
    bg_color = scene.background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training")
    first_iter += 1
    physics_losses = {}

    for iteration in progress_bar:
        iter_start.record()

        # Determine phase
        if iteration <= opt.phase1_iterations:
            phase = "phase1"
        elif iteration <= opt.phase3_start:
            phase = "phase2"
        else:
            phase = "phase3"

        # Get dynamic weights for current phase
        lambda_weights = get_dynamic_lambda_weights(iteration, opt, phase)

        # --------------------------------------------------------------
        # Phase-specific sampling
        # MLP is always trainable (like 4DGS deformation — never frozen).
        # Phase 1 uses t=0 → MLP naturally learns identity (Δ≈0), smooth Phase 2 entry.
        # --------------------------------------------------------------
        if phase == "phase1":
            if len(t0_cameras) == 0:
                raise RuntimeError("No t=0 cameras available for Phase 1!")
            viewpoint_cam = t0_cameras[iteration % len(t0_cameras)]
        else:
            viewpoint_cam = train_cameras[random.randint(0, len(train_cameras) - 1)]

        # --------------------------------------------------------------
        # Forward pass: render with spatio-temporal deformation
        # --------------------------------------------------------------
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg_color,
            is_training=True
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"]
        )

        # --------------------------------------------------------------
        # Visual losses
        # --------------------------------------------------------------
        gt_image = viewpoint_cam.gt_image.cuda()[:3, :, :]
        Ll1 = l1_loss(image, gt_image)

        # SSIM loss
        ssim_loss_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = lambda_weights['lambda_rgb'] * Ll1 + \
            lambda_weights['lambda_ssim'] * (1.0 - ssim_loss_val)

        # --------------------------------------------------------------
        # Physics-based losses (on DEFORMED + ACTIVATED state)
        # Weights (λ) control per-phase magnitude, so always compute.
        # Density (KNN) is strided internally for performance.
        # --------------------------------------------------------------
        physics_losses = gaussians.compute_physics_losses(
            p_t=render_pkg["deformed_xyz"],
            scales_activated=render_pkg["scales_activated"],
            opacities_activated=render_pkg["opacities_activated"],
            shs=render_pkg["shs"],
            iteration=iteration,
            lambda_weights=lambda_weights,
        )
        for loss_name, loss_val in physics_losses.items():
            loss += loss_val

        loss.backward()

        # Gradient clipping: stabilise MLP + transform feature training
        torch.nn.utils.clip_grad_norm_(
            gaussians.spatio_temporal_encoder.parameters(),
            max_norm=1.0
        )
        torch.nn.utils.clip_grad_norm_(
            [gaussians._transform_feature],
            max_norm=1.0
        )

        iter_end.record()

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "Phase": phase,
                    "N": gaussians._xyz.shape[0],
                })

            if iteration % 100 == 0:
                # Log to TensorBoard
                tb_writer.add_scalar('train/loss', loss.item(), iteration)
                tb_writer.add_scalar('train/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train/ssim', ssim_loss_val.item(), iteration)
                tb_writer.add_scalar('train/num_points', gaussians._xyz.shape[0], iteration)
                tb_writer.add_scalar('train/phase',
                                     0 if phase == "phase1" else (1 if phase == "phase2" else 2),
                                     iteration)

                # Log physics losses
                for loss_name, loss_val in physics_losses.items():
                    tb_writer.add_scalar(f'train/{loss_name}', loss_val.item(), iteration)

                # PSNR
                with torch.no_grad():
                    psnr_val = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
                    tb_writer.add_scalar('train/psnr', psnr_val.item(), iteration)

        # --------------------------------------------------------------
        # Densification (adaptive control)
        # --------------------------------------------------------------
        if iteration < opt.densify_until_iter:
            # Update densification stats
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and \
               iteration % opt.densification_interval == 0:
                # Average gradient over accumulated steps
                avg_grad = gaussians.xyz_gradient_accum / \
                    gaussians.denom.clamp(min=1)
                # Clone small Gaussians with high gradient
                gaussians.densify_and_clone(
                    avg_grad,
                    opt.densify_grad_threshold,
                    scene.cameras_extent,
                )
                # Split large Gaussians with high gradient
                gaussians.densify_and_split(
                    avg_grad,
                    opt.densify_grad_threshold,
                    scene.cameras_extent,
                )

                # Prune low-opacity or too-large Gaussians
                prune_mask = (gaussians.opacity_activation(
                    gaussians._opacity) < opt.opacity_threshold).squeeze()
                if prune_mask.any():
                    gaussians.prune_points(prune_mask)

                # Reset accumulated gradient stats
                torch.cuda.empty_cache()

            if iteration > 0 and (iteration % opt.opacity_reset_interval == 0 or
               (dataset.white_background and iteration == opt.densify_from_iter)):
                gaussians.reset_opacity()

        # --------------------------------------------------------------
        # Optimizer step
        # --------------------------------------------------------------
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------------
        if iteration in checkpoint_iterations:
            print(f"\n[ITER {iteration}] Saving checkpoint...")
            checkpoint_dir = os.path.join(
                dataset.model_path, "point_cloud", f"iteration_{iteration}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            gaussians.save_ply(os.path.join(checkpoint_dir, "point_cloud.ply"))
            gaussians.save_deformation(checkpoint_dir)

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------
        if iteration in testing_iterations:
            print(f"\n[ITER {iteration}] Running validation...")
            validate(gaussians, scene, pipe, bg_color, iteration, tb_writer)

    print("\nTraining complete!")
    return gaussians


def validate(gaussians, scene, pipe, bg_color, iteration, tb_writer):
    """Validation: render test views and compute metrics."""
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("  No test cameras, skipping validation")
        return

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    for viewpoint in test_cameras:
        render_pkg = render(
            viewpoint, gaussians, pipe, bg_color,
            is_training=False
        )
        image = render_pkg["render"]
        gt_image = viewpoint.gt_image.cuda()[:3, :, :]

        psnr_val = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
        ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

        psnr_total += psnr_val
        ssim_total += ssim_val
        count += 1

    avg_psnr = psnr_total / count if count > 0 else 0
    avg_ssim = ssim_total / count if count > 0 else 0

    print(f"  Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

    if tb_writer is not None:
        tb_writer.add_scalar('val/psnr', avg_psnr, iteration)
        tb_writer.add_scalar('val/ssim', avg_ssim, iteration)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaussFluids Training")

    # Parameter groups
    lp = ModelParams()
    pp = PipelineParams()
    gp = GaussFluidsParams()
    op = OptimizationParams()

    lp.add_arguments(parser)
    pp.add_arguments(parser)
    gp.add_arguments(parser)
    op.add_arguments(parser)

    args = parser.parse_args(sys.argv[1:])

    # Assign values from args
    lp.sh_degree = args.sh_degree
    lp.source_path = args.source_path
    lp.model_path = args.model_path
    lp.white_background = args.white_background
    lp.resolution = args.resolution
    lp.eval = args.eval
    lp.load_iteration = args.load_iteration

    pp.convert_SHs_python = args.convert_SHs_python
    pp.compute_cov3D_python = args.compute_cov3D_python
    pp.debug = args.debug

    gp.transform_feature_dim = args.transform_feature_dim
    gp.mlp_hidden_dim = args.mlp_hidden_dim
    gp.mlp_num_layers = args.mlp_num_layers
    gp.smoothing_length = args.smoothing_length
    gp.knn_k = args.knn_k
    gp.time_pe_freqs = args.time_pe_freqs
    gp.physics_loss_stride = args.physics_loss_stride

    for attr_name in ['iterations', 'phase1_iterations', 'phase2_start', 'phase3_start',
                      'lambda_rgb', 'lambda_ssim', 'lambda_dens', 'lambda_aniso',
                      'lambda_vol', 'lambda_op', 'lambda_light',
                      'position_lr_init', 'feature_lr', 'opacity_lr', 'scaling_lr',
                      'rotation_lr', 'transform_feature_lr', 'mlp_lr',
                      'percent_dense', 'densify_from_iter', 'densify_until_iter',
                      'densification_interval', 'opacity_reset_interval',
                      'densify_grad_threshold', 'opacity_threshold',
                      'batch_size', 'dataloader']:
        if hasattr(args, attr_name):
            setattr(op, attr_name, getattr(args, attr_name))

    # Create output directories
    os.makedirs(lp.model_path, exist_ok=True)
    print(f"Output directory: {lp.model_path}")

    # Setup
    safe_state(False)

    # Checkpoint / test iterations
    testing_iterations = list(range(0, op.iterations + 1, 2000))
    saving_iterations = list(range(0, op.iterations + 1, 7000))
    checkpoint_iterations = list(range(0, op.iterations + 1, 7000))
    if op.iterations not in saving_iterations:
        saving_iterations.append(op.iterations)
    if op.iterations not in checkpoint_iterations:
        checkpoint_iterations.append(op.iterations)

    debug_from = -1

    # Train
    training(lp, op, pp, gp, testing_iterations, saving_iterations,
             checkpoint_iterations, debug_from)

    print("All done!")

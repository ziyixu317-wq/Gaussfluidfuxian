#
# GaussFluids: Configuration Parameters
# Paper Table 1 weights, 3-phase schedule, hyperparameters
#

import argparse


class ModelParams:
    """Scene and model parameters."""
    def __init__(self):
        self.sh_degree = 1           # GaussFluids: 1st-order SH only (paper Sec 3.1)
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = True
        self.eval = True
        self.extension = ".png"
        self.load_iteration = None

    def add_arguments(self, parser):
        parser.add_argument('--sh_degree', type=int, default=1,
                          help='Spherical harmonics degree (GaussFluids: 1)')
        parser.add_argument('--source_path', type=str, required=True,
                          help='Path to dataset directory')
        parser.add_argument('--model_path', type=str, required=True,
                          help='Path to output directory')
        parser.add_argument('--images', type=str, default='images')
        parser.add_argument('--resolution', type=int, default=-1)
        parser.add_argument('--white_background', action='store_true', default=True)
        parser.add_argument('--eval', action='store_true', default=True)
        parser.add_argument('--extension', type=str, default='.png')
        parser.add_argument('--add_points', action='store_true', default=False)
        parser.add_argument('--load_iteration', type=int, default=None,
                          help='Load checkpoint at specified iteration')


class PipelineParams:
    """Rendering pipeline parameters."""
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

    def add_arguments(self, parser):
        parser.add_argument('--convert_SHs_python', action='store_true', default=False)
        parser.add_argument('--compute_cov3D_python', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=False)


class GaussFluidsParams:
    """
    GaussFluids-specific hyperparameters.
    Paper values: transform_feature_dim=64, mlp_hidden=256, mlp_layers=4,
                  smoothing_length=0.3, knn_k=64
    """
    def __init__(self):
        self.transform_feature_dim = 64      # Paper: 64
        self.mlp_hidden_dim = 256            # Paper: 256
        self.mlp_num_layers = 4              # Paper: 4
        self.smoothing_length = 0.3          # Paper: 0.3
        self.knn_k = 64                      # Paper: 64
        self.time_pe_freqs = 10              # NeRF PE frequencies for time
        self.physics_loss_stride = 5         # Compute physics every N iterations

    def add_arguments(self, parser):
        parser.add_argument('--transform_feature_dim', type=int, default=64)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256)
        parser.add_argument('--mlp_num_layers', type=int, default=4)
        parser.add_argument('--smoothing_length', type=float, default=0.3)
        parser.add_argument('--knn_k', type=int, default=64)
        parser.add_argument('--time_pe_freqs', type=int, default=10)
        parser.add_argument('--physics_loss_stride', type=int, default=5)


class OptimizationParams:
    """
    Training optimization and scheduling parameters.
    Includes corrected loss weights from paper Table 1.
    """
    def __init__(self):
        # Iteration counts for 3 phases
        self.iterations = 14000              # Total iterations
        self.phase1_iterations = 2000        # Canonical frame phase
        self.phase2_start = 2000             # Dynamics phase starts
        self.phase3_start = 8000             # Refinement phase starts

        # Loss weights (Paper Table 1, CORRECTED)
        self.lambda_rgb = 0.2       # L1 visual loss
        self.lambda_ssim = 0.1      # SSIM visual loss
        self.lambda_dens = 0.1      # Density (incompressibility)
        self.lambda_aniso = 0.1     # Anisotropy constraint
        self.lambda_vol = 0.1       # Volume consistency
        self.lambda_op = 0.01       # Opacity consistency (CORRECTED: 0.01 not 1.0)
        self.lambda_light = 1.0     # Light/SH consistency

        # Learning rates
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.transform_feature_lr = 0.01     # Paper: 1e-2
        self.mlp_lr = 0.0001                 # Paper: 1e-4
        self.percent_dense = 0.01

        # Densification
        self.densify_from_iter = 500
        self.densify_until_iter = 8000
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_grad_threshold = 0.0002
        self.opacity_threshold = 0.005

        # Batch
        self.batch_size = 1
        self.dataloader = False

    def add_arguments(self, parser):
        parser.add_argument('--iterations', type=int, default=14000)
        parser.add_argument('--phase1_iterations', type=int, default=2000)
        parser.add_argument('--phase2_start', type=int, default=2000)
        parser.add_argument('--phase3_start', type=int, default=8000)

        # Loss weights
        parser.add_argument('--lambda_rgb', type=float, default=0.2)
        parser.add_argument('--lambda_ssim', type=float, default=0.1)
        parser.add_argument('--lambda_dens', type=float, default=0.1)
        parser.add_argument('--lambda_aniso', type=float, default=0.1)
        parser.add_argument('--lambda_vol', type=float, default=0.1)
        parser.add_argument('--lambda_op', type=float, default=0.01)
        parser.add_argument('--lambda_light', type=float, default=1.0)

        # LR
        parser.add_argument('--position_lr_init', type=float, default=0.00016)
        parser.add_argument('--position_lr_final', type=float, default=0.0000016)
        parser.add_argument('--feature_lr', type=float, default=0.0025)
        parser.add_argument('--opacity_lr', type=float, default=0.05)
        parser.add_argument('--scaling_lr', type=float, default=0.005)
        parser.add_argument('--rotation_lr', type=float, default=0.001)
        parser.add_argument('--transform_feature_lr', type=float, default=0.01)
        parser.add_argument('--mlp_lr', type=float, default=0.0001)
        parser.add_argument('--percent_dense', type=float, default=0.01)

        # Densification
        parser.add_argument('--densify_from_iter', type=int, default=500)
        parser.add_argument('--densify_until_iter', type=int, default=8000)
        parser.add_argument('--densification_interval', type=int, default=100)
        parser.add_argument('--opacity_reset_interval', type=int, default=3000)
        parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
        parser.add_argument('--opacity_threshold', type=float, default=0.005)

        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--dataloader', action='store_true', default=False)

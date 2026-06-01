#
# GaussFluids: Scene Class
# Loads data, initializes GaussianFluidParticles, manages camera lists
#

import os
import numpy as np
import torch
from random import randint

from scene.gaussian_fluid_particles import GaussianFluidParticles
from scene.dataset_readers import (
    readCamerasFromTransforms, readPoints3D,
    cameraList_from_camInfos
)
from utils.graphics_utils import BasicPointCloud


class Scene:
    """
    GaussFluids scene: loads dataset, initializes or loads model,
    manages train/test camera lists.
    """

    def __init__(self, args, gaussians: GaussianFluidParticles,
                 load_ply_path=None, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """
        Args:
            args: ModelParams (from arguments)
            gaussians: GaussianFluidParticles model instance
            load_ply_path: optional path to load PLY checkpoint
            load_iteration: optional iteration number for checkpoint
            shuffle: shuffle training cameras
            resolution_scales: list of resolution scales
        """
        self.gaussians = gaussians
        self.args = args

        # Detect dataset type and load
        transforms_path = os.path.join(args.source_path, "transforms_train.json")

        if os.path.exists(transforms_path):
            print(f"Found transforms_train.json, loading 4DGS_data format from {args.source_path}")
            train_cam_infos = readCamerasFromTransforms(
                args.source_path, "transforms_train.json",
                args.white_background
            )
            # Try loading test cameras
            test_transforms = os.path.join(args.source_path, "transforms_test.json")
            if os.path.exists(test_transforms):
                test_cam_infos = readCamerasFromTransforms(
                    args.source_path, "transforms_test.json",
                    args.white_background
                )
                print(f"Loaded {len(test_cam_infos)} test cameras")
            else:
                test_cam_infos = []
        else:
            raise FileNotFoundError(
                f"No transforms_train.json found in {args.source_path}. "
                f"GaussFluids requires 4DGS_data format."
            )

        print(f"Loaded {len(train_cam_infos)} training cameras")

        # Get time range
        all_times = sorted(set([c.time for c in train_cam_infos]))
        self.maxtime = max(all_times) if all_times else 1.0
        self.mintime = min(all_times) if all_times else 0.0
        print(f"Time range: [{self.mintime:.3f}, {self.maxtime:.3f}], "
              f"{len(all_times)} unique timestamps")

        # Separate t=0 cameras for Phase 1
        self.t0_cam_infos = [c for c in train_cam_infos if c.time <= self.mintime + 1e-6]
        print(f"Phase 1 (t=0) cameras: {len(self.t0_cam_infos)}")

        # Convert to Camera objects
        resolution_scale = resolution_scales[0] if resolution_scales else 1.0
        self.train_cameras = cameraList_from_camInfos(train_cam_infos, resolution_scale, args)
        self.test_cameras = cameraList_from_camInfos(test_cam_infos, resolution_scale, args)
        self.t0_cameras = cameraList_from_camInfos(self.t0_cam_infos, resolution_scale, args)

        # Compute scene extent
        cam_centers = []
        for cam in self.train_cameras:
            cam_centers.append(cam.camera_center.cpu().numpy())
        cam_centers = np.stack(cam_centers)
        self.cameras_extent = np.linalg.norm(
            cam_centers.max(axis=0) - cam_centers.min(axis=0)
        )
        print(f"Scene extent: {self.cameras_extent:.3f}")

        # Initialize or load model
        if load_ply_path is not None and os.path.exists(load_ply_path):
            print(f"Loading Gaussians from {load_ply_path}")
            gaussians.load_ply(load_ply_path)
        elif load_iteration is not None:
            ply_path = os.path.join(args.model_path, "point_cloud",
                                    f"iteration_{load_iteration}", "point_cloud.ply")
            print(f"Loading Gaussians from {ply_path}")
            gaussians.load_ply(ply_path)

            # Load deformation
            deform_path = os.path.join(args.model_path, "point_cloud",
                                       f"iteration_{load_iteration}")
            if os.path.exists(os.path.join(deform_path, "spatio_temporal_encoder.pth")):
                gaussians.load_deformation(deform_path)
        else:
            # Initialize from point cloud
            print("Initializing Gaussian particles...")
            self._initialize_from_point_cloud(gaussians, args, train_cam_infos)

        # Setup random background
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def _initialize_from_point_cloud(self, gaussians, args, cam_infos):
        """Initialize Gaussian particles from COLMAP point cloud or random init."""
        # Try loading COLMAP point cloud
        ply_path = os.path.join(args.source_path, "points3D.ply")
        sparse_path = os.path.join(args.source_path, "sparse", "0", "points3D.ply")

        pcd = None
        for path in [ply_path, sparse_path]:
            if os.path.exists(path):
                print(f"Loading point cloud from {path}")
                points, colors, normals = readPoints3D(path)
                pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
                break

        if pcd is None:
            # Random initialization: estimate scene center from camera positions,
            # then place particles in a tight Gaussian around center.
            # Paper uses COLMAP init; without it we need dense particles in
            # the expected fluid region for convergence.
            print("No COLMAP point cloud found, initializing random point cloud...")
            cam_positions = []
            for cam_info in cam_infos[:min(50, len(cam_infos))]:
                w2c = np.linalg.inv(np.vstack([
                    np.hstack([cam_info.R.T, cam_info.T.reshape(3, 1)]),
                    [0, 0, 0, 1]
                ]))
                cam_positions.append(w2c[:3, 3])

            cam_positions = np.stack(cam_positions)
            center = cam_positions.mean(axis=0)
            extent = np.linalg.norm(cam_positions.max(axis=0) - cam_positions.min(axis=0))

            # Dense particles in tight central region (fluid is near center)
            num_pts = 50000
            radius = extent * 0.03  # ~0.65 units for extent=21.7
            points = center + (np.random.randn(num_pts, 3)) * radius
            # Initialize colors based on camera view colors
            colors = np.ones_like(points) * 0.5  # neutral gray
            normals = np.zeros_like(points)
            normals[:, 2] = 1.0  # default normal pointing up
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

            print(f"Initialized {num_pts} random points within extent {extent:.3f}")

        gaussians.create_from_pcd(pcd, self.cameras_extent)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def getT0Cameras(self):
        """Get t=0 cameras for Phase 1 training."""
        return self.t0_cameras

    def getTrainCamerasByTime(self, time):
        """Get training cameras at a specific time."""
        return [cam for cam in self.train_cameras
                if abs(cam.time - time) < 1e-4]

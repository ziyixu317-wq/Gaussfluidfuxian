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
        self.cameras_extent = min(
            np.linalg.norm(cam_centers.max(axis=0) - cam_centers.min(axis=0)),
            5.0  # cap: fluid is small, large extent over-scales position_lr
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
            # No COLMAP — generate pseudo point cloud from multi-view images.
            # Use alpha channel of t=0 frames to identify foreground, project
            # rays into 3D, and initialize particles near the visual hull.
            print("No COLMAP point cloud found, generating from multi-view images...")
            pcd = self._init_from_images(args, cam_infos)
            if pcd is None:
                # Ultimate fallback
                print("Image-based init failed, using random initialization...")
                num_pts = 20000
                center = np.zeros(3)
                points = center + (np.random.randn(num_pts, 3)) * 1.0
                colors = np.ones_like(points) * 0.5
                normals = np.zeros_like(points)
                normals[:, 2] = 1.0
                pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
            print(f"Initialized {pcd.points.shape[0]} points")

        gaussians.create_from_pcd(pcd, self.cameras_extent)

    def _init_from_images(self, args, cam_infos):
        """
        Generate pseudo point cloud from multi-view t=0 images.
        Uses foreground pixels (alpha channel) to place particles near the visual hull.
        """
        from PIL import Image
        t0_infos = [c for c in cam_infos if c.time <= self.mintime + 1e-6]
        if len(t0_infos) < 2:
            return None
        print(f"  Using {len(t0_infos)} t=0 camera views for initialization")

        all_points = []
        all_colors = []
        pts_per_view = 2000  # sample per camera view

        for cam_info in t0_infos:
            try:
                img = Image.open(cam_info.image_path)
                img_np = np.array(img.convert("RGBA")) / 255.0
            except Exception:
                continue

            alpha = img_np[:, :, 3]
            fg_mask = alpha > 0.1
            fg_ys, fg_xs = np.where(fg_mask)
            if len(fg_ys) < 10:
                continue

            # Sample foreground pixels
            n_sample = min(pts_per_view, len(fg_ys))
            idx = np.random.choice(len(fg_ys), n_sample, replace=False)
            px_x = fg_xs[idx]
            px_y = fg_ys[idx]
            colors_sample = img_np[px_y, px_x, :3]

            # Build camera-to-world transform
            R_c2w = cam_info.R  # cam_info.R is w2c[:3, :3].T = R_c2w
            cam_center = -R_c2w @ cam_info.T

            # Compute ray directions for sampled pixels
            h, w = img_np.shape[:2]
            focal = (w / 2) / np.tan(cam_info.FovX / 2)

            # Normalized device coordinates (COLMAP convention)
            ndc_x = (px_x - w / 2) / focal
            ndc_y = (px_y - h / 2) / focal
            ndc_z = np.ones_like(ndc_x)

            dirs_cam = np.stack([ndc_x, ndc_y, ndc_z], axis=-1)
            dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

            # Rotate to world
            dirs_world = (R_c2w @ dirs_cam.T).T

            # Estimate depth range: camera distance to origin, fluid is near center.
            # Sample depths uniformly to fill a volume (not a shell).
            cam_dist = np.linalg.norm(cam_center)
            depth_min = cam_dist * 0.3
            depth_max = cam_dist * 0.8
            depths = np.random.uniform(depth_min, depth_max, size=len(dirs_world))
            pts = cam_center + dirs_world * depths[:, np.newaxis]

            all_points.append(pts)
            all_colors.append(colors_sample)

        if len(all_points) == 0:
            return None

        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0

        print(f"  Generated {points.shape[0]} points from foreground pixels")
        return BasicPointCloud(points=points, colors=colors, normals=normals)

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

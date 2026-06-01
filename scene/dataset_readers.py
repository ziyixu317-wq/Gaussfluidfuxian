#
# GaussFluids: Dataset Readers
# Supports 4DGS_data transforms JSON format (NeRF-style)
# Also supports COLMAP format for point cloud initialization
#

import os
import sys
import json
import math
import numpy as np
import torch
from typing import NamedTuple
from PIL import Image
from utils.graphics_utils import fov2focal as _fov2focal


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    mask: np.array


# ---------------------------------------------------------------------------
# Transforms JSON reader (4DGS_data format)
# ---------------------------------------------------------------------------

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """
    Read camera info from NeRF-style transforms JSON.
    Compatible with 4DGS_data format.

    JSON format:
    {
        "camera_angle_x": float (horizontal FOV in radians),
        "frames": [
            {
                "file_path": "./train/r_121_000",
                "rotation": 0.0,
                "time": 0.0,
                "transform_matrix": [[...], ...]  # 4×4 camera-to-world
            },
            ...
        ]
    }
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]

    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        # Build image path (file_path omits extension in 4DGS_data)
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_name = os.path.basename(frame["file_path"]) + extension

        # Load image
        image = Image.open(cam_name)

        # Get time
        time = float(frame.get("time", 0.0))

        # Parse transform matrix (camera-to-world)
        c2w = np.array(frame["transform_matrix"])

        # Convert from NeRF convention (camera-to-world) to 3DGS convention
        # In NeRF: c2w = [R | t] maps camera coords to world coords
        # In 3DGS: R = transpose of rotation, T is translation
        # Following 3DGS convention:
        # Transform matrix is camera-to-world, we need world-to-camera for R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T  # R is stored transposed in 3DGS
        T = w2c[:3, 3]

        # Handle image loading: keep alpha as foreground mask
        im_data = np.array(image.convert("RGBA")) / 255.0
        alpha = im_data[:, :, 3]  # foreground mask: 1=fluid, 0=background
        bg = np.array([1.0, 1.0, 1.0]) if white_background else np.array([0.0, 0.0, 0.0])

        # Composite onto background
        arr = im_data[:, :, :3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
        image_tensor = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

        # Also create alpha mask as a PIL image for Camera
        alpha_mask = Image.fromarray((alpha * 255.0).astype(np.uint8), "L")

        width, height = image_tensor.size
        fovy = _focal2fov(_fov2focal(fovx, width), height)

        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
            image=image_tensor, image_path=cam_name,
            image_name=image_name, width=width, height=height,
            time=time, mask=alpha_mask
        ))

    return cam_infos


def _focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# ---------------------------------------------------------------------------
# COLMAP reader (for point cloud initialization)
# ---------------------------------------------------------------------------

def readPoints3D(path):
    """Read COLMAP points3D.ply for initial point cloud."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions, colors, normals


# ---------------------------------------------------------------------------
# Camera info helpers for Scene class
# ---------------------------------------------------------------------------

def camera_to_JSON(camera_infos):
    """Serialize camera infos for checkpointing."""
    json_cams = []
    for cam in camera_infos:
        json_cams.append({
            'id': cam.uid,
            'img_name': cam.image_name,
            'width': cam.width,
            'height': cam.height,
            'time': float(cam.time),
        })
    return json_cams


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """Convert CameraInfo list to Camera objects."""
    from scene.camera import Camera
    camera_list = []

    for c in cam_infos:
        image_tensor = torch.from_numpy(np.array(c.image)) / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)

        # Convert alpha mask to tensor
        if c.mask is not None:
            alpha_tensor = torch.from_numpy(np.array(c.mask)).float() / 255.0
        else:
            alpha_tensor = None

        if resolution_scale != 1.0:
            import torch.nn.functional as F
            new_h = int(c.height * resolution_scale)
            new_w = int(c.width * resolution_scale)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear'
            ).squeeze(0)
            if alpha_tensor is not None:
                alpha_tensor = F.interpolate(
                    alpha_tensor.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w), mode='nearest'
                ).squeeze()

        camera_list.append(Camera(
            colmap_id=c.uid, R=c.R, T=c.T,
            FoVx=c.FovX, FoVy=c.FovY,
            image=image_tensor,
            gt_alpha_mask=alpha_tensor,
            image_name=c.image_name,
            uid=c.uid,
            time=c.time,
        ))

    return camera_list

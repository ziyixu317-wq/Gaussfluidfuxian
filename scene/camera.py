#
# GaussFluids: Camera class (adapted from 3DGS)
#

import torch
import math
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:
    """Camera class for GaussFluids rendering."""

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, time=0.0, trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device="cuda"):
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.image_name = image_name
        self.uid = uid
        self.time = time
        self.data_device = data_device

        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        if gt_alpha_mask is not None:
            self.gt_image = torch.cat(
                (image[:3], gt_alpha_mask.unsqueeze(0)), dim=0
            )
        else:
            self.gt_image = image

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """Lightweight camera for deferred rendering."""

    def __init__(self, width, height, fovy, fovx, znear, zfar,
                 world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = world_view_transform.inverse()
        self.camera_center = view_inv[3, :3]
        self.time = time

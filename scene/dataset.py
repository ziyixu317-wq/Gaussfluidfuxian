#
# GaussFluids: PyTorch Dataset wrapper
#

import torch
from torch.utils.data import Dataset


class FourDGSdataset(Dataset):
    """
    Dataset wrapper for multi-view temporal data.
    Returns camera views with images for training.
    """

    def __init__(self, cameras, resolution_scale=1.0):
        self.cameras = cameras
        self.resolution_scale = resolution_scale
        self.count = len(cameras)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        cam = self.cameras[idx]
        return {
            'image': cam.gt_image,
            'R': cam.R,
            'T': cam.T,
            'FoVx': cam.FoVx,
            'FoVy': cam.FoVy,
            'time': cam.time,
            'uid': cam.uid,
            'image_name': cam.image_name,
        }


class CameraDataset(Dataset):
    """Simple camera-only dataset for rendering."""

    def __init__(self, cameras):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import rotate, resize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from network import Network


class CamLocDataset(Dataset):
    """Camera localization dataset providing access to image, calibration,
    and ground truth data given a dataset directory."""

    def __init__(self, root_dir,
                 mode=1,
                 sparse=True,
                 augment=False,
                 aug_rotation=30,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_contrast=0.1,
                 aug_brightness=0.1,
                 image_height=448):

        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)
        self.image_height = image_height

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_contrast = aug_contrast
        self.aug_brightness = aug_brightness

        if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            print("WARNING: Augmentation is applied but coordinates will not be augmented.")

        rgb_dir = root_dir + '/rgb/'
        if self.eye:
            coord_dir = root_dir + '/eye/'
        elif self.sparse:
            coord_dir = root_dir + '/init/'
        else:
            coord_dir = root_dir + '/depth/'

        self.rgb_files = sorted([rgb_dir + f for f in os.listdir(rgb_dir)])

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_height),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        if self.init or self.eye:
            self.coord_files = sorted([coord_dir + f for f in os.listdir(coord_dir)])

        if not sparse:
            # Create a grid of 2D pixel positions when generating scene coordinates from depth
            self.prediction_grid = np.zeros((2,
                                             math.ceil(5000 / Network.OUTPUT_SUBSAMPLE),
                                             math.ceil(5000 / Network.OUTPUT_SUBSAMPLE)))
            for x in range(0, self.prediction_grid.shape[2]):
                for y in range(0, self.prediction_grid.shape[1]):
                    self.prediction_grid[0, y, x] = x * Network.OUTPUT_SUBSAMPLE
                    self.prediction_grid[1, y, x] = y * Network.OUTPUT_SUBSAMPLE

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        image = io.imread(self.rgb_files[idx])
        pose = 0

        if len(image.shape) < 3:
            image2 = color.gray2rgb(image)
        if len(image.shape) == 3:
            image2 = color.rgb2gray(image)

        mask = self.generate_mask(image2)
        image2, mask = self.resize_with_mask(image2, mask, self.image_height)

        if self.init:
            if self.sparse:
                coords = torch.load(self.coord_files[idx])
                coords[0] /= 3500
                coords[1] /= 3500
            else:
                depth = io.imread(self.coord_files[idx]).astype(np.float64)
                depth /= 1000  # mm to meters
        elif self.eye:
            coords = torch.load(self.coord_files[idx])
        else:
            coords = 0

        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            cur_image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(self.image_height * scale_factor)),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_brightness, contrast=self.aug_contrast),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4], std=[0.25])
            ])

            image = cur_image_transform(image)

            def my_rot(t, angle, order, mode='constant'):
                t = t.permute(1, 2, 0).numpy()
                t = rotate(t, angle, order=order, mode=mode)
                t = torch.from_numpy(t).permute(2, 0, 1).float()
                return t

            image = my_rot(image, angle, 1, 'reflect')

            if self.init:
                if self.sparse:
                    coords_w = math.ceil(image.size(2) / Network.OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(image.size(1) / Network.OUTPUT_SUBSAMPLE)
                    coords = my_rot(coords, angle, 0)
                else:
                    depth = resize(depth, image.shape[1:], order=0)
                    depth = rotate(depth, angle, order=0, mode='constant')

            angle = angle * math.pi / 180
            pose = 0
        else:
            image = self.image_transform(image)
            coords_w = math.ceil(image.size(2) / Network.OUTPUT_SUBSAMPLE)
            coords_h = math.ceil(image.size(1) / Network.OUTPUT_SUBSAMPLE)
            coords = coords.float()
            coords = F.interpolate(coords.unsqueeze(0), size=(coords_h, coords_w))[0]

        if self.init and not self.sparse:
            # Generate initialization targets from depth map
            offsetX = int(Network.OUTPUT_SUBSAMPLE / 2)
            offsetY = int(Network.OUTPUT_SUBSAMPLE / 2)

            coords = torch.zeros((
                3,
                math.ceil(image.shape[1] / Network.OUTPUT_SUBSAMPLE),
                math.ceil(image.shape[2] / Network.OUTPUT_SUBSAMPLE)))

            depth = depth[offsetY::Network.OUTPUT_SUBSAMPLE, offsetX::Network.OUTPUT_SUBSAMPLE]

            xy = self.prediction_grid[:, :depth.shape[0], :depth.shape[1]].copy()
            xy[0] += offsetX
            xy[1] += offsetY
            xy[0] -= image.shape[2] / 2
            xy[1] -= image.shape[1] / 2
            xy /= 1e-6 + 1  # placeholder to prevent div-by-zero if focal_length=0
            xy[0] *= depth
            xy[1] *= depth

            eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
            eye[0:2] = xy
            eye[2] = depth
            eye[3] = 1

            sc = np.matmul(pose, eye.reshape(4, -1))
            sc = sc.reshape(4, depth.shape[0], depth.shape[1])
            sc[:, depth == 0] = 0
            sc[:, depth > 1000] = 0
            sc = torch.from_numpy(sc[0:3])
            coords[:, :sc.shape[1], :sc.shape[2]] = sc

        return image, pose, coords, mask, self.rgb_files[idx]

    def generate_mask(self, image):
        """Generate a mask identifying valid regions (non-black pixels)."""
        mask = (image > 0)
        return mask

    def resize_with_mask(self, image, mask, target_height):
        """Resize both image and mask to target height while keeping aspect ratio."""
        h, w = image.shape
        scale_factor = target_height / h
        target_width = int(w * scale_factor)
        image_resized = resize(image, (target_height, target_width), anti_aliasing=True)
        mask_resized = resize(mask.astype(np.float32), (target_height, target_width), anti_aliasing=True)
        return image_resized, mask_resized

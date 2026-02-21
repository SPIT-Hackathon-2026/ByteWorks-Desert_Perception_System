"""
Dataset with strong augmentation via albumentations.

Key differences from previous pipeline:
  - 512×512 resolution (up from 476×266)
  - Heavy spatial + colour augmentation during training
  - Proper synchronized image+mask augmentation
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from umixformer_pipeline.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from umixformer_pipeline.utils import convert_mask


# ============================================================================
# Augmentation pipelines
# ============================================================================

def get_train_augmentations(size: int = IMG_SIZE) -> A.Compose:
    """Heavy augmentation pipeline for training."""
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0),
                            ratio=(0.75, 1.333), interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.25),
        A.Affine(shift_limit=0.05, scale_limit=0.1,
                 rotate_limit=15, border_mode=0, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10,
                                  sat_shift_limit=20,
                                  val_shift_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.3),
        ], p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                        hole_width_range=(8, 32), fill="random", p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_augmentations(size: int = IMG_SIZE) -> A.Compose:
    """Minimal augmentation for validation: resize + normalise."""
    return A.Compose([
        A.Resize(height=size, width=size, interpolation=1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================================
# Dataset
# ============================================================================

class OffroadSegDataset(Dataset):
    """Paired image/mask dataset with albumentations augmentation.

    Folder layout::

        data_dir/
            Color_Images/
                0001.png ...
            Segmentation/
                0001.png ...
    """

    def __init__(self, data_dir: str, augmentations=None, return_filename: bool = False):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.augmentations = augmentations
        self.return_filename = return_filename
        self.ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        image = np.array(Image.open(
            os.path.join(self.image_dir, fname)).convert("RGB"))
        mask_pil = Image.open(os.path.join(self.mask_dir, fname))
        mask = np.array(convert_mask(mask_pil), dtype=np.int64)

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed["image"]       # (C, H, W) float tensor
            mask = transformed["mask"]          # (H, W) int64 tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        mask = mask.long()

        if self.return_filename:
            return image, mask, fname
        return image, mask


# ============================================================================
# DataLoader factory
# ============================================================================

def build_train_loader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    ds = OffroadSegDataset(data_dir, augmentations=get_train_augmentations())
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def build_val_loader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    ds = OffroadSegDataset(data_dir, augmentations=get_val_augmentations())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def build_test_loader(data_dir: str, batch_size: int = 1, num_workers: int = 2) -> DataLoader:
    ds = OffroadSegDataset(data_dir, augmentations=get_val_augmentations(),
                           return_filename=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

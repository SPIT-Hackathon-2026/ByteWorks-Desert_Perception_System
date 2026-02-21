"""
Dataset and transform definitions for the off-road segmentation pipeline.
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from offroad_training_pipeline.config import IMG_H, IMG_W, IMAGENET_MEAN, IMAGENET_STD
from offroad_training_pipeline.utils import convert_mask


# ============================================================================
# Transforms
# ============================================================================

def get_image_transform(h: int = IMG_H, w: int = IMG_W) -> T.Compose:
    """Standard image transform: resize → tensor → ImageNet normalise."""
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_mask_transform(h: int = IMG_H, w: int = IMG_W) -> T.Compose:
    """Mask transform: NEAREST resize (preserve class IDs!) → tensor."""
    return T.Compose([
        T.Resize((h, w), interpolation=InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    """Paired colour-image / segmentation-mask dataset.

    Folder layout expected::

        data_dir/
            Color_Images/
                0001.png
                ...
            Segmentation/
                0001.png
                ...

    Parameters
    ----------
    data_dir : str
        Root that contains ``Color_Images/`` and ``Segmentation/``.
    transform, mask_transform : callable, optional
        Applied to the PIL images before returning.
    return_filename : bool
        If *True* each sample is ``(image, mask, filename)``.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        mask_transform=None,
        return_filename: bool = False,
    ):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_filename = return_filename
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255  # back to class ids

        if self.return_filename:
            return image, mask, data_id
        return image, mask


# ============================================================================
# DataLoader factory
# ============================================================================

def build_dataloader(
    data_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    return_filename: bool = False,
) -> DataLoader:
    """Convenience builder: creates dataset + dataloader in one call."""
    transform = get_image_transform()
    mask_transform = get_mask_transform()
    ds = MaskDataset(
        data_dir=data_dir,
        transform=transform,
        mask_transform=mask_transform,
        return_filename=return_filename,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

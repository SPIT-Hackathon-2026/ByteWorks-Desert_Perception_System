"""
Batch colorisation of segmentation masks.

Reads single-channel masks, maps unique values to random colours, and writes
RGB PNGs.

Usage
-----
    python -m offroad_training_pipeline.visualize
    python -m offroad_training_pipeline.visualize --input_dir path/to/masks
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from offroad_training_pipeline.config import DATASET_ROOT


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Colorise segmentation masks")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset"),
        help="Folder containing mask images",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output folder (default: <input_dir>/colorized)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "colorized")
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        f for f in Path(input_dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Found {len(image_files)} image(s) to process")

    color_map: dict[int, np.ndarray] = {}

    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        im = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"  Skipped (unreadable): {img_file.name}")
            continue

        im2 = np.zeros((*im.shape[:2], 3), dtype=np.uint8)
        for v in np.unique(im):
            if v not in color_map:
                color_map[v] = np.random.randint(0, 255, (3,), dtype=np.uint8)
            im2[im == v] = color_map[v]

        out_path = os.path.join(output_dir, f"{img_file.stem}.png")
        cv2.imwrite(out_path, im2)
        print(f"  Saved: {out_path}")

    print(f"\nDone – {len(image_files)} images → {output_dir}/")
    print(f"Unique values encountered: {len(color_map)}")


if __name__ == "__main__":
    main()

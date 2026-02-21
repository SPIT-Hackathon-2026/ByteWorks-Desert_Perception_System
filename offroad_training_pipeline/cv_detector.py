"""
OpenCV-based object detector for rare terrain classes.

The neural network struggles with rare classes (Logs ≈0.06 %, Flowers ≈2 %,
Dry Bushes ≈1.6 %, Rocks ≈5 %).  This module learns HSV colour profiles from
labelled training data and uses histogram back-projection + morphological
filtering to detect those classes at inference time.

The results are merged with NN predictions in the test pipeline via
``merge_predictions()``.
"""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from offroad_training_pipeline.config import (
    CLASS_NAMES,
    CV_CLASSES,
    TRAIN_DIR,
    VALUE_MAP,
)


class CVObjectDetector:
    """Classical-CV detector for rare object classes.

    Workflow
    --------
    1.  ``learn_profiles(train_dir)`` — scans training masks, collects HSV
        pixels for each CV class, builds 2-D Hue–Saturation histograms.
    2.  ``detect(image_rgb_uint8)`` — runs histogram back-projection +
        morphological clean-up + connected-component filtering.
    3.  ``merge_predictions(...)`` — combines NN + CV outputs into a single
        10-class segmentation map.
    """

    def __init__(self):
        self.histograms: dict[int, np.ndarray] = {}
        self.hsv_ranges: dict[int, tuple] = {}
        self._learned = False

        # Per-class detection parameters (tuned for off-road scenes)
        self._params: dict[int, dict] = {
            3: {"min_area": 60, "morph_k": 5, "bp_thresh": 35},   # Dry Bushes
            5: {"min_area": 20, "morph_k": 3, "bp_thresh": 40},   # Flowers
            6: {"min_area": 30, "morph_k": 3, "bp_thresh": 30},   # Logs
            7: {"min_area": 80, "morph_k": 5, "bp_thresh": 35},   # Rocks
        }

    # ------------------------------------------------------------------ #
    #  Learning                                                           #
    # ------------------------------------------------------------------ #

    def learn_profiles(
        self, train_dir: str | None = None, max_samples: int = 500
    ) -> None:
        """Learn HSV colour histograms from training images + masks."""
        if train_dir is None:
            train_dir = TRAIN_DIR

        img_dir = os.path.join(train_dir, "Color_Images")
        mask_dir = os.path.join(train_dir, "Segmentation")

        files = sorted(os.listdir(img_dir))[:max_samples]
        print(f"\n{'=' * 60}")
        print(f"Learning CV colour profiles from {len(files)} training images …")
        print(f"{'=' * 60}")

        class_pixels: dict[int, list[np.ndarray]] = {c: [] for c in CV_CLASSES}

        for fname in tqdm(files, desc="Scanning", leave=False):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            raw_mask = np.array(Image.open(mask_path))
            class_mask = np.zeros(raw_mask.shape[:2], dtype=np.uint8)
            for raw_val, class_id in VALUE_MAP.items():
                class_mask[raw_mask == raw_val] = class_id

            for cid in CV_CLASSES:
                roi = class_mask == cid
                if roi.any():
                    px = img_hsv[roi]
                    # Sub-sample large regions to keep memory reasonable
                    if len(px) > 3000:
                        idx = np.random.choice(len(px), 3000, replace=False)
                        px = px[idx]
                    class_pixels[cid].append(px)

        print("\nCV class colour profiles:")
        for cid in CV_CLASSES:
            if not class_pixels[cid]:
                print(
                    f"  {CLASS_NAMES[cid]:<20}: "
                    "⚠  No pixels found in training data"
                )
                continue

            all_px = np.concatenate(class_pixels[cid])
            n = len(all_px)

            # 2-D Hue–Saturation histogram for back-projection
            hsv_roi = all_px.reshape(-1, 1, 3).astype(np.uint8)
            hist = cv2.calcHist(
                [hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.histograms[cid] = hist

            # Percentile-based HSV ranges (secondary signal)
            lower = np.percentile(all_px, 2, axis=0).astype(np.uint8)
            upper = np.percentile(all_px, 98, axis=0).astype(np.uint8)
            self.hsv_ranges[cid] = (lower, upper)

            print(
                f"  {CLASS_NAMES[cid]:<20}: {n:>8,} px  "
                f"HSV [{lower[0]:3d},{lower[1]:3d},{lower[2]:3d}]"
                f" → [{upper[0]:3d},{upper[1]:3d},{upper[2]:3d}]"
            )

        self._learned = True
        print("CV profiles learned ✓\n")

    # ------------------------------------------------------------------ #
    #  Detection                                                          #
    # ------------------------------------------------------------------ #

    def detect(
        self, image_rgb_uint8: np.ndarray
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Detect CV classes in an RGB uint8 image.

        Returns
        -------
        masks : dict[int, ndarray[bool]]
            Binary detection mask per CV class.
        confidences : dict[int, ndarray[float32]]
            Pixel-wise confidence (0–1) per CV class.
        """
        assert self._learned, "Call learn_profiles() first"

        img_bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = image_rgb_uint8.shape[:2]

        masks: dict[int, np.ndarray] = {}
        confidences: dict[int, np.ndarray] = {}

        for cid in CV_CLASSES:
            if cid not in self.histograms:
                masks[cid] = np.zeros((h, w), dtype=bool)
                confidences[cid] = np.zeros((h, w), dtype=np.float32)
                continue

            params = self._params.get(
                cid, {"min_area": 50, "morph_k": 5, "bp_thresh": 40}
            )

            # ---- histogram back-projection ----
            bp = cv2.calcBackProject(
                [img_hsv], [0, 1],
                self.histograms[cid],
                [0, 180, 0, 256], 1,
            )
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bp = cv2.filter2D(bp, -1, disc)

            # Confidence map [0, 1]
            confidences[cid] = bp.astype(np.float32) / 255.0

            # ---- HSV range thresholding (secondary) ----
            lower, upper = self.hsv_ranges[cid]
            range_mask = cv2.inRange(img_hsv, lower, upper)

            # Combine: back-projection AND range
            combined = cv2.bitwise_and(bp, range_mask)

            # Threshold
            _, mask = cv2.threshold(
                combined, params["bp_thresh"], 255, cv2.THRESH_BINARY
            )

            # ---- morphological clean-up ----
            k = params["morph_k"]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # ---- remove small connected components ----
            n_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            min_area = params["min_area"]
            clean = np.zeros_like(mask)
            for lid in range(1, n_labels):
                if stats[lid, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels_cc == lid] = 255

            masks[cid] = clean > 0

        return masks, confidences

    # ------------------------------------------------------------------ #
    #  Merging NN + CV                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def merge_predictions(
        nn_pred: np.ndarray,
        nn_conf: np.ndarray,
        cv_masks: dict[int, np.ndarray],
        cv_confidences: dict[int, np.ndarray],
        nn_conf_threshold: float = 0.6,
    ) -> np.ndarray:
        """Merge NN predictions with CV detections.

        Parameters
        ----------
        nn_pred : (H, W) int
            NN class predictions (only NN-class IDs after masking CV channels).
        nn_conf : (H, W) float
            NN max softmax probability.
        cv_masks : {class_id: (H, W) bool}
            CV binary detection masks.
        cv_confidences : {class_id: (H, W) float32}
            CV per-pixel confidence maps.
        nn_conf_threshold : float
            Below this NN confidence, CV may override.

        Returns
        -------
        final : (H, W) int — merged 10-class prediction.
        """
        final = nn_pred.copy()

        # Apply CV classes in priority order (most distinctive first)
        priority = [6, 5, 7, 3]  # Logs, Flowers, Rocks, Dry Bushes

        for cv_cid in priority:
            if cv_cid not in cv_masks:
                continue
            detected = cv_masks[cv_cid]
            if not detected.any():
                continue

            cv_conf = cv_confidences.get(
                cv_cid, np.ones_like(nn_conf, dtype=np.float32)
            )

            # Override where CV detected AND
            #   (NN is uncertain  OR  CV confidence is very high)
            override = detected & (
                (nn_conf < nn_conf_threshold) | (cv_conf > 0.7)
            )
            final[override] = cv_cid

        return final

    # ------------------------------------------------------------------ #
    #  CV-specific metrics                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_cv_metrics(
        cv_masks: dict[int, np.ndarray],
        gt_labels: np.ndarray,
        cv_classes: list[int],
    ) -> dict:
        """Compute precision / recall / F1 for CV detections.

        Parameters
        ----------
        cv_masks : {class_id: (H, W) bool}
        gt_labels : (H, W) int — ground-truth class IDs
        cv_classes : list of CV class indices

        Returns
        -------
        results : dict   Per-class and average metrics.
        """
        results: dict = {}
        precisions, recalls, f1s = [], [], []

        for cid in cv_classes:
            pred = cv_masks.get(cid, np.zeros_like(gt_labels, dtype=bool))
            gt = gt_labels == cid

            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[cid] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

            if gt.any():  # only average over classes present in GT
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        results["avg_precision"] = (
            float(np.mean(precisions)) if precisions else 0.0
        )
        results["avg_recall"] = float(np.mean(recalls)) if recalls else 0.0
        results["avg_f1"] = float(np.mean(f1s)) if f1s else 0.0

        return results

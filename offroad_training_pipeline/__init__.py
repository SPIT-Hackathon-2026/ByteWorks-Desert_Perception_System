"""
offroad_training_pipeline
=========================

Modular off-road image segmentation pipeline built on DINOv2 features.

Quick start::

    # Train
    python -m offroad_training_pipeline.train --model convnext_head --epochs 10

    # Test / infer
    python -m offroad_training_pipeline.test --model convnext_head

    # Colorise raw masks
    python -m offroad_training_pipeline.visualize
"""

from offroad_training_pipeline.config import NUM_CLASSES, CLASS_NAMES  # noqa: F401

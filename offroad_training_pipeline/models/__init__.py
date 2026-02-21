"""
Models sub-package.

Provides:
* ``MODEL_REGISTRY`` – dict mapping name → class
* ``build_model(name, **kwargs)`` – factory function
* ``load_backbone(size, device)`` – loads a frozen DINOv2 backbone
"""

from offroad_training_pipeline.models.registry import MODEL_REGISTRY, build_model
from offroad_training_pipeline.models.backbone import load_backbone

# Register all heads so they appear in the registry at import time
import offroad_training_pipeline.models.convnext_head       # noqa: F401
import offroad_training_pipeline.models.convnext_deep_head  # noqa: F401
import offroad_training_pipeline.models.hybrid_head         # noqa: F401
import offroad_training_pipeline.models.linear_head         # noqa: F401
import offroad_training_pipeline.models.mlp_head            # noqa: F401
import offroad_training_pipeline.models.multiscale_head     # noqa: F401
import offroad_training_pipeline.models.segformer_head      # noqa: F401

__all__ = ["MODEL_REGISTRY", "build_model", "load_backbone"]

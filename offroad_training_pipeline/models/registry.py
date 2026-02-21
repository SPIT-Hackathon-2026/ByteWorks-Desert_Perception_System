"""
Model registry – a thin dict-based registry so you can add new segmentation
heads by simply decorating them with ``@register_model("my_head")``.

Usage
-----
>>> from offroad_training_pipeline.models.registry import register_model
>>> @register_model("my_custom_head")
... class MyHead(nn.Module): ...
"""

from __future__ import annotations
from typing import Dict, Type
from torch import nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Class decorator that adds a model to ``MODEL_REGISTRY``."""
    def _decorator(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return _decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by *name* with arbitrary keyword args."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)

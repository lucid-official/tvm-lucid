"""TVM sized for 32x32 CIFAR-10 in pixel space.

The method lives in :mod:`tvm_lucid.tvm`; this module only pins what the
data dictates — three channels, 32 pixels, ten classes — and lets the
config's ``[model]`` section override every other :class:`TVMConfig` field.
"""

from typing import Any

from tvm_lucid.tvm import TVM, TVMConfig

DEVICE = "metal"

# What CIFAR-10 in pixel space requires, whatever the config says.
_CIFAR10 = {"sample_size": 32, "in_channels": 3, "out_channels": 3, "num_classes": 10}


def make_config(model_overrides: dict[str, Any]) -> TVMConfig:
    overrides = dict(model_overrides)
    for key in _CIFAR10:
        if key in overrides:
            raise KeyError(f"[model].{key} is fixed by the CIFAR-10 setup")
    if isinstance(overrides.get("guidance_range"), list):
        overrides["guidance_range"] = tuple(overrides["guidance_range"])
    return TVMConfig(**_CIFAR10, **overrides)


def build(model_overrides: dict[str, Any]) -> TVM:
    """A fresh, randomly initialised TVM on the Metal device, in train mode."""
    return TVM(make_config(model_overrides)).to(DEVICE).train()


def count_parameters(model: TVM) -> int:
    return sum(p.numel() for p in model.parameters())

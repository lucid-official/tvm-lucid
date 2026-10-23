"""The paper's TVM from Lucid's model zoo, re-sized for 32x32 CIFAR-10.

Nothing here re-implements the method: the network, the objective and the
sampler are ``lucid.models.generative.tvm``.  This module only fixes what a
pixel-space CIFAR-10 run needs — three channels, ten classes, no latent
whitening — and lets ``[model]`` in the config override everything else.
"""

from typing import Any

import lucid.models as M

DEVICE = "metal"

# What CIFAR-10 in pixel space requires, whatever the config says.
_CIFAR10 = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "num_classes": 10,
    "latent_mean": (0.0, 0.0, 0.0),
    "latent_std": (1.0, 1.0, 1.0),
}


def build(model_overrides: dict[str, Any]) -> M.TVMForImageGeneration:
    """A fresh, randomly initialised TVM on the Metal device, in train mode."""
    overrides = dict(model_overrides)
    for key in _CIFAR10:
        if key in overrides:
            raise KeyError(f"[model].{key} is fixed by the CIFAR-10 setup")
    model = M.tvm_imagenet256_gen(**_CIFAR10, **overrides)
    return model.to(DEVICE).train()


def count_parameters(model: M.TVMForImageGeneration) -> int:
    return sum(p.numel() for p in model.parameters())

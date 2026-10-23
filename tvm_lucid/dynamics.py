"""Training-dynamics probes: what the loss alone does not show.

Each quantity is cheap on its own but forces a GPU read-back, so the
heavier ones are taken every ``train.dynamics_every`` steps rather than
every step.

* **Gradient and parameter norms by group** — ``embed`` (patch, time,
  interval, guidance, label), each ``blocks.i``, and ``final``.  A block
  whose gradient collapses or explodes relative to its neighbours shows up
  here long before it shows up in the loss.
* **Update-to-weight ratio** :math:`\\lVert\\Delta\\theta\\rVert /
  \\lVert\\theta\\rVert` per step — the effective learning rate Adam is
  actually applying.
* **EMA drift** :math:`\\lVert\\theta^* - \\theta\\rVert / \\lVert\\theta\\rVert`
  and the same for :math:`\\theta^{**}` — how far the target and evaluation
  networks lag the live one.
* **Activation norms** of the time embedding and of the last block's raw
  adaLN modulation — the quantity the paper plots in Figure 4 to show
  that DiT without Lipschitz control blows up and TVM's does not.
"""

import math
import re
from collections import defaultdict
from typing import Any

import lucid
import lucid.models as M
import lucid.nn as nn
from lucid import Tensor

_BLOCK = re.compile(r"^tvm\.blocks\.(\d+)\.")


def group_of(name: str) -> str:
    """The reporting group of a parameter name."""
    match = _BLOCK.match(name)
    if match:
        return f"blocks.{int(match.group(1)):02d}"
    if name.startswith("tvm.final."):
        return "final"
    return "embed"


def _sq(t: Tensor) -> Tensor:
    return (t.detach().float() ** 2).sum()


def grouped_norms(model: M.TVMForImageGeneration) -> dict[str, dict[str, float]]:
    """Per-group and global L2 norms of parameters and gradients."""
    params: dict[str, list[Tensor]] = defaultdict(list)
    grads: dict[str, list[Tensor]] = defaultdict(list)
    for name, param in model.named_parameters():
        group = group_of(name)
        params[group].append(_sq(param))
        if param.grad is not None:
            grads[group].append(_sq(param.grad))
    out: dict[str, dict[str, float]] = {"param": {}, "grad": {}}
    total_p = total_g = 0.0
    for group in sorted(params):
        p2 = float(sum(params[group][1:], params[group][0]).item())
        g2 = float(sum(grads[group][1:], grads[group][0]).item()) if grads[group] else 0.0
        out["param"][group] = math.sqrt(p2)
        out["grad"][group] = math.sqrt(g2)
        total_p += p2
        total_g += g2
    out["param"]["total"] = math.sqrt(total_p)
    out["grad"]["total"] = math.sqrt(total_g)
    return out


def snapshot(model: M.TVMForImageGeneration) -> dict[str, Tensor]:
    """A detached copy of the live weights, to measure one step's update."""
    with lucid.no_grad():
        return {n: p.detach().clone() for n, p in model.tvm.named_parameters()}


def _relative_distance(a: dict[str, Tensor], b: dict[str, Tensor]) -> float:
    with lucid.no_grad():
        diff = sum(float(_sq(a[n] - b[n]).item()) for n in a)
        base = sum(float(_sq(b[n]).item()) for n in a)
    return math.sqrt(diff / max(base, 1e-30))


def update_ratio(before: dict[str, Tensor], model: M.TVMForImageGeneration) -> float:
    """``||θ_after − θ_before|| / ||θ_before||``."""
    after = {n: p for n, p in model.tvm.named_parameters()}
    return _relative_distance(after, before)


def ema_drift(model: M.TVMForImageGeneration) -> dict[str, float]:
    """How far each EMA sits from the live weights, relative to their size."""
    live = {n: p for n, p in model.tvm.named_parameters()}
    out: dict[str, float] = {}
    if model.config.target_ema_decay > 0.0:
        out["target"] = _relative_distance(dict(model.target_network.named_parameters()), live)
    if model.config.eval_ema_decay is not None:
        out["eval"] = _relative_distance(dict(model.ema_network.named_parameters()), live)
    return out


class ActivationProbe:
    """RMS of chosen modules' outputs, collected only while armed.

    The hooks fire on every forward — the JVP's, the Flow Matching term's —
    so arming them only on dynamics steps keeps the cost off ordinary steps.
    """

    def __init__(self, modules: dict[str, nn.Module]) -> None:
        self.armed = False
        self._values: dict[str, list[Tensor]] = defaultdict(list)
        self._handles = [
            module.register_forward_hook(self._hook(name)) for name, module in modules.items()
        ]

    def _hook(self, name: str) -> Any:
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if self.armed and isinstance(output, Tensor):
                self._values[name].append((output.detach().float() ** 2).mean().sqrt())

        return record

    def read(self) -> dict[str, float]:
        """Mean RMS per module since the last read; disarms."""
        out = {
            name: sum(float(v.item()) for v in values) / len(values)
            for name, values in self._values.items()
            if values
        }
        self._values.clear()
        self.armed = False
        return out

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def activation_probe(model: M.TVMForImageGeneration) -> ActivationProbe:
    """Figure 4's quantities: the time embedding and the last raw modulation."""
    net = model.tvm
    return ActivationProbe(
        {
            "time_embed": net.time_embed,
            "adaln_last": net.blocks[len(net.blocks) - 1].ada_ln,
            "adaln_final": net.final.ada_ln,
        }
    )

"""Where a training step's time goes, phase by phase, on Metal.

    python scripts/profile_step.py --config configs/cifar10_s2.toml
    python scripts/profile_step.py --config configs/cifar10_s2.toml --set train.batch_size=32

Metal is lazy: an op runs only when something downstream is evaluated, so a
plain timer around the step measures graph building, not work.  Here every
phase of the step is forced (``lucid.eval`` + ``metal.synchronize``) and timed
on its own, averaged over ``--steps`` steps after ``--warmup``.  The phases
mirror ``TVM.forward`` (same draws, same order); class dropout is skipped.

It then runs whole ``Trainer.step`` calls under cProfile and prints the
host-side hot spots: time inside eval/synchronize is the GPU working; time
anywhere else is Python dispatching ops while the GPU waits.
"""

import argparse
import cProfile
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import lucid  # noqa: E402
import lucid.metal as metal  # noqa: E402

from tvm_lucid import config as config_mod  # noqa: E402
from tvm_lucid.model import DEVICE  # noqa: E402
from tvm_lucid.train import Trainer  # noqa: E402


def timed_step(trainer: Trainer, images, labels) -> dict[str, float]:  # type: ignore[no-untyped-def]
    model, c = trainer.model, trainer.model.config
    batch = int(images.shape[0])
    out: dict[str, float] = {}
    clock = [time.perf_counter()]

    def mark(name: str, *tensors: object) -> None:
        lucid.eval(*tensors)
        metal.synchronize()
        now = time.perf_counter()
        out[name] = now - clock[0]
        clock[0] = now

    t, s, s_fm = model.sample_times(batch, DEVICE)
    guidance = model.sample_guidance(batch, DEVICE)
    null = lucid.full((batch,), c.num_classes, device=DEVICE, dtype=lucid.int64)
    noise = lucid.randn(tuple(int(d) for d in images.shape), device=DEVICE)
    t_b, s_b, w_b = t.reshape(-1, 1, 1, 1), s_fm.reshape(-1, 1, 1, 1), guidance.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_b) * images + t_b * noise
    x_fm = (1.0 - s_b) * images + s_b * noise
    mark("draws", x_t, x_fm)

    velocity, along_s = model.net.jvp_s(x_t, t, s, labels, guidance)
    terminal = velocity + (s - t).reshape(-1, 1, 1, 1) * along_s
    mark("jvp_s: live net + s-tangent", velocity, terminal)

    landing = (x_t + (s - t).reshape(-1, 1, 1, 1) * velocity).detach()
    with lucid.no_grad():
        reads = model.target_network(
            lucid.cat([landing, x_fm], dim=0),
            lucid.cat([s, s_fm], dim=0),
            lucid.cat([s, s_fm], dim=0),
            lucid.cat([labels, null], dim=0),
            lucid.cat([guidance, lucid.ones_like(guidance)], dim=0),
        )
    mark("target read: theta*, 2B, no grad", reads)

    velocity_fm = model.net(x_fm, s_fm, s_fm, labels, guidance)
    mark("flow-matching forward", velocity_fm)

    guided = w_b * (noise - images) + (1.0 - w_b) * reads[batch:]
    differs = 1.0 - lucid.isclose(s, t).float()
    term = (((terminal - reads[:batch]) / w_b) ** 2).mean(dim=(1, 2, 3)) * differs
    fm = (((velocity_fm - guided) / w_b) ** 2).mean(dim=(1, 2, 3))
    loss = (term + fm).mean()
    mark("loss", loss)

    loss.backward()
    grads = [p.grad for p in model.net.parameters() if p.grad is not None]
    mark("backward", *grads)

    trainer.opt.step()
    trainer.opt.zero_grad()
    mark("AdamW", *model.net.parameters())

    model.update_ema()
    mark("both EMAs")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    cfg = config_mod.load(args.config, args.set)
    trainer = Trainer(cfg, Path("/tmp"))
    batches = trainer.loader.epoch(0)

    def batch():  # type: ignore[no-untyped-def]
        images, labels = next(batches)
        return lucid.tensor(images, device=DEVICE), lucid.tensor(labels, device=DEVICE)

    for _ in range(args.warmup):
        trainer.step([next(batches)], dynamics=False)

    totals: dict[str, float] = defaultdict(float)
    for _ in range(args.steps):
        for name, seconds in timed_step(trainer, *batch()).items():
            totals[name] += seconds / args.steps
    whole = sum(totals.values())
    print(f"\nbatch {cfg.train.batch_size}: {whole:.2f} s per step, phase by phase")
    for name, seconds in totals.items():
        print(f"  {name:<36s} {seconds:8.3f} s  {100 * seconds / whole:5.1f}%")

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    for _ in range(args.steps):
        trainer.step([next(batches)], dynamics=False)
    metal.synchronize()
    profiler.disable()
    print(f"\nTrainer.step: {(time.perf_counter() - started) / args.steps:.2f} s per step")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()

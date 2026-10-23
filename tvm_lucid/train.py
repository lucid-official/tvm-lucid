"""Train TVM on CIFAR-10: ``python -m tvm_lucid.train --config configs/cifar10_s2.toml``.

Algorithm 1 of the paper, one optimiser step at a time: the objective
(``TVMForImageGeneration.forward``), the optimiser, then both EMAs
(``update_ema``).  Around it: deterministic resumable data, tqdm bars,
JSONL logs of the training dynamics, end-of-epoch evaluation, atomic
checkpoints, a wall-clock budget, and a guard against non-finite losses.

    # measure (≈2 min), then decide batch size / budget
    python -m tvm_lucid.train --config configs/cifar10_s2.toml --benchmark
    # train (new run directory under runs/)
    python -m tvm_lucid.train --config configs/cifar10_s2.toml
    # resume a run where it stopped (optionally changing the budget)
    python -m tvm_lucid.train --resume runs/cifar10_s2-20261024-010203 --set train.hours=24
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import lucid
import lucid.metal as metal
import lucid.optim
import numpy as np
from lucid.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tvm_lucid import config as config_mod
from tvm_lucid.data import EpochLoader, load_split
from tvm_lucid.dynamics import activation_probe, ema_drift, grouped_norms, snapshot, update_ratio
from tvm_lucid.evaluate import Evaluator
from tvm_lucid.model import DEVICE, build, count_parameters
from tvm_lucid.run import (
    JsonlLog,
    environment,
    load_checkpoint,
    new_run_dir,
    save_checkpoint,
    write_json,
)

GiB = 2**30


def _reseed(seed: int, global_step: int) -> None:
    """The model's own draws (times, noise, label dropout) for this point.

    Metal's generator state cannot be saved, so a resumed run reseeds from
    ``(seed, step)``: deterministic, though not the stream an uninterrupted
    run would have drawn.  The data order is exact either way.
    """
    mixed = (seed * 1_000_003 + global_step) % (2**31 - 1)
    lucid.manual_seed(mixed)
    metal.manual_seed(mixed)


class Trainer:
    def __init__(self, cfg: config_mod.ExperimentConfig, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        root = config_mod.resolve_root(cfg)
        self.train_x, self.train_y = load_split(root, "train")
        self.test_x, _ = load_split(root, "test")
        tc = cfg.train
        self.loader = EpochLoader(
            self.train_x, self.train_y, tc.batch_size, cfg.seed, cfg.data.flip
        )
        _reseed(cfg.seed, 0)
        self.model = build(cfg.model)
        self.opt = lucid.optim.AdamW(
            self.model.parameters(),
            lr=tc.lr,
            betas=tuple(tc.betas),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )
        self.epoch = 0
        self.step_in_epoch = 0  # in loader batches
        self.global_step = 0  # in optimiser steps
        self.best_fid = math.inf
        self.nonfinite_streak = 0
        self.notes: dict[str, Any] = {}
        # Hooks stay registered; they record only while armed (dynamics steps).
        self.probe = activation_probe(self.model)

    # ── checkpoints ──────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        m = self.model
        state: dict[str, Any] = {
            "model": m.state_dict(),
            "opt": self.opt.state_dict(),
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "best_fid": self.best_fid if math.isfinite(self.best_fid) else -1.0,
            "config": self.cfg.to_dict(),
        }
        if m.config.target_ema_decay > 0.0:
            state["target"] = m.target_network.state_dict()
        if m.config.eval_ema_decay is not None:
            state["ema"] = m.ema_network.state_dict()
        return state

    def restore(self, path: Path) -> None:
        state = load_checkpoint(path)
        m = self.model
        m.load_state_dict(state["model"])
        if "target" in state:
            m.target_network.load_state_dict(state["target"])
        if "ema" in state:
            m.ema_network.load_state_dict(state["ema"])
        self.opt.load_state_dict(state["opt"])
        self.epoch = int(state["epoch"])
        self.step_in_epoch = int(state["step_in_epoch"])
        self.global_step = int(state["global_step"])
        best = float(state["best_fid"])
        self.best_fid = best if best >= 0 else math.inf
        _reseed(self.cfg.seed, self.global_step)

    def checkpoint(self, name: str) -> None:
        save_checkpoint(self.run_dir / "ckpt" / name, self.state())

    # ── one optimiser step ───────────────────────────────────────────────

    def step(self, micro: list[tuple[np.ndarray, np.ndarray]], dynamics: bool) -> dict[str, Any]:
        """Objective → (clip) → AdamW → both EMAs, over ``len(micro)`` batches."""
        tc = self.cfg.train
        model = self.model
        if dynamics:
            self.probe.armed = True
            before = snapshot(model)
        started = time.perf_counter()
        totals = {"loss": 0.0, "terminal": 0.0, "fm": 0.0}
        for images, labels in micro:
            out = model(lucid.tensor(images, device=DEVICE), lucid.tensor(labels, device=DEVICE))
            (out.loss / len(micro)).backward()
            # One scalar read per micro-batch: it is also the non-finite guard,
            # and it lets the step time below mean something.
            totals["loss"] += float(out.loss.item()) / len(micro)
            totals["terminal"] += float(out.terminal_loss.item()) / len(micro)
            totals["fm"] += float(out.flow_matching_loss.item()) / len(micro)

        record: dict[str, Any] = dict(totals)
        if not math.isfinite(totals["loss"]):
            self.opt.zero_grad()
            self.nonfinite_streak += 1
            if dynamics:
                self.probe.read()
            record["skipped"] = True
            return record
        self.nonfinite_streak = 0

        if tc.grad_clip > 0:
            record["grad_norm_preclip"] = float(clip_grad_norm_(model.parameters(), tc.grad_clip).item())
        if dynamics:
            norms = grouped_norms(model)
        self.opt.step()
        self.opt.zero_grad()
        model.update_ema()
        metal.synchronize()
        record["step_s"] = time.perf_counter() - started

        if dynamics:
            record["dynamics"] = {
                "grad_norm": norms["grad"],
                "param_norm": norms["param"],
                "update_ratio": update_ratio(before, model),
                "ema_drift": ema_drift(model),
                "activation_rms": self.probe.read(),
            }
        return record

    # ── the loop ─────────────────────────────────────────────────────────

    def fit(self, evaluator: Evaluator | None) -> None:
        tc = self.cfg.train
        logs = {
            name: JsonlLog(self.run_dir / f"{name}.jsonl")
            for name in ("metrics", "epoch", "events")
        }
        per_epoch = self.loader.steps_per_epoch
        images_per_step = tc.batch_size * tc.grad_accum
        deadline = time.time() + tc.hours * 3600
        logs["events"].write(
            {"event": "start", "global_step": self.global_step, "epoch": self.epoch,
             "params": count_parameters(self.model), "hours": tc.hours, **self.notes}
        )  # fmt: skip

        stop_reason = None
        epochs = tqdm(
            total=tc.epochs, initial=self.epoch, desc="epochs", position=0, dynamic_ncols=True
        )
        try:
            while self.epoch < tc.epochs and stop_reason is None:
                bar = tqdm(
                    total=per_epoch,
                    initial=self.step_in_epoch,
                    desc=f"epoch {self.epoch}",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    unit="batch",
                )
                micro: list[tuple[np.ndarray, np.ndarray]] = []
                window, window_steps = time.perf_counter(), 0
                for images, labels in self.loader.epoch(self.epoch, start=self.step_in_epoch):
                    micro.append((images, labels))
                    self.step_in_epoch += 1
                    bar.update(1)
                    if len(micro) < tc.grad_accum:
                        continue
                    dynamics = self.global_step % tc.dynamics_every == 0
                    record = self.step(micro, dynamics)
                    micro = []
                    if record.get("skipped"):
                        logs["events"].write({"event": "nonfinite_skip", "global_step": self.global_step, **record})
                        if self.nonfinite_streak >= tc.max_nonfinite:
                            stop_reason = f"{self.nonfinite_streak} non-finite losses in a row"
                            break
                        continue
                    self.global_step += 1
                    window_steps += 1

                    if self.global_step % tc.log_every == 0 or dynamics:
                        elapsed = time.perf_counter() - window
                        rate = images_per_step * window_steps / max(elapsed, 1e-9)
                        window, window_steps = time.perf_counter(), 0
                        dyn = record.pop("dynamics", None)
                        row = {
                            "kind": "step",
                            "global_step": self.global_step,
                            "epoch": self.epoch + self.step_in_epoch / per_epoch,
                            **record,
                            "lr": tc.lr,
                            "img_per_s": rate,
                            "mem_gib": metal.memory_allocated() / GiB,
                            "peak_gib": metal.max_memory_allocated() / GiB,
                        }
                        logs["metrics"].write(row)
                        if dyn is not None:
                            logs["metrics"].write({"kind": "dynamics", "global_step": self.global_step, **dyn})
                        bar.set_postfix(
                            loss=f"{record['loss']:.4f}",
                            term=f"{record['terminal']:.4f}",
                            fm=f"{record['fm']:.4f}",
                            ips=f"{row['img_per_s']:.0f}",
                            mem=f"{row['peak_gib']:.1f}G",
                        )
                    if time.time() >= deadline:
                        stop_reason = f"time budget of {tc.hours} h reached"
                        break
                bar.close()
                if stop_reason is not None:
                    break

                # epoch complete
                finished = self.epoch
                self.epoch += 1
                self.step_in_epoch = 0
                self.checkpoint("last.lucid")
                if tc.keep_every_epochs and (finished + 1) % tc.keep_every_epochs == 0:
                    self.checkpoint(f"epoch_{finished:04d}.lucid")
                row = {"epoch": finished, "global_step": self.global_step}
                if evaluator is not None:
                    row.update(self.evaluate(evaluator, finished, logs["events"]))
                logs["epoch"].write(row)
                epochs.update(1)
                shown = {k: f"{v:.2f}" for k, v in row.items() if k.startswith("fid_nfe")}
                epochs.set_postfix(step=self.global_step, **shown)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        finally:
            epochs.close()
            self.checkpoint("last.lucid")
            logs["events"].write(
                {"event": "stop", "reason": stop_reason or "all epochs done",
                 "global_step": self.global_step, "epoch": self.epoch,
                 "step_in_epoch": self.step_in_epoch}
            )  # fmt: skip
            for log in logs.values():
                log.close()
            tqdm.write(f"stopped: {stop_reason or 'all epochs done'} — {self.run_dir}")

    def evaluate(self, evaluator: Evaluator, epoch: int, events: JsonlLog) -> dict[str, Any]:
        ec = self.cfg.eval
        row: dict[str, Any] = {}
        started = time.time()
        if ec.every_epochs and epoch % ec.every_epochs == 0:
            row.update(evaluator.grids(self.model, f"epoch_{epoch:04d}"))
            row.update(evaluator.probe(self.model))
        if ec.fid_every_epochs and (epoch + 1) % ec.fid_every_epochs == 0:
            row.update(evaluator.fid(self.model))
            first = f"fid_nfe{ec.fid_nfe[0]}"
            if row[first] < self.best_fid:
                self.best_fid = row[first]
                self.checkpoint("best_fid.lucid")
                events.write({"event": "best_fid", "epoch": epoch, first: row[first]})
        row["eval_seconds"] = round(time.time() - started, 1)
        return row

    # ── benchmark ────────────────────────────────────────────────────────

    def benchmark(self, steps: int = 20, warmup: int = 5) -> None:
        tc = self.cfg.train
        batches = self.loader.epoch(0)
        for index in tqdm(range(warmup + steps), desc="benchmark"):
            if index == warmup:
                metal.synchronize()
                metal.reset_peak_memory_stats()
                started = time.perf_counter()
            micro = [next(batches) for _ in range(tc.grad_accum)]
            self.step(micro, dynamics=False)
        metal.synchronize()
        seconds = (time.perf_counter() - started) / steps
        per_step = tc.batch_size * tc.grad_accum
        budget = tc.hours * 3600 / seconds * per_step
        epoch_s = self.loader.steps_per_epoch / tc.grad_accum * seconds
        print(json.dumps({
            "params_M": round(count_parameters(self.model) / 1e6, 2),
            "batch": tc.batch_size, "grad_accum": tc.grad_accum,
            "s_per_step": round(seconds, 3),
            "ms_per_image": round(seconds / per_step * 1e3, 2),
            "peak_gib": round(metal.max_memory_allocated() / GiB, 2),
            "minutes_per_epoch": round(epoch_s / 60, 1),
            "epochs_in_budget": round(budget / len(self.train_x), 1),
            "hours": tc.hours,
        }, indent=2))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TVM on CIFAR-10 with Lucid.")
    parser.add_argument("--config", type=Path, help="TOML experiment file")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="where new runs go")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="skip end-of-epoch evaluation")
    args = parser.parse_args()

    if args.resume is not None:
        stored = json.loads((args.resume / "config.json").read_text())
        cfg = config_mod.load(args.config, args.set, base=stored)
        run_dir = args.resume
    else:
        if args.config is None:
            parser.error("--config is required for a new run")
        cfg = config_mod.load(args.config, args.set)
        run_dir = None

    if args.benchmark:
        trainer = Trainer(cfg, Path("/tmp"))
        trainer.benchmark()
        return

    if run_dir is None:
        run_dir = new_run_dir(args.runs, cfg.name)
        write_json(run_dir / "env.json", environment())
    write_json(run_dir / "config.json", cfg.to_dict())

    trainer = Trainer(cfg, run_dir)
    if args.resume is not None:
        trainer.restore(run_dir / "ckpt" / "last.lucid")
        trainer.notes = {"resumed": True, "overrides": args.set}
        tqdm.write(f"resumed {run_dir} at epoch {trainer.epoch}, step {trainer.global_step}")
    tqdm.write(f"run directory: {run_dir}  ({count_parameters(trainer.model) / 1e6:.1f}M params)")

    evaluator = None
    if not args.no_eval:
        evaluator = Evaluator(
            cfg.eval,
            run_dir,
            trainer.train_x,
            trainer.test_x,
            config_mod.resolve_root(cfg) / "cifar10_train_inception_stats.npz",
        )
    trainer.fit(evaluator)


if __name__ == "__main__":
    main()

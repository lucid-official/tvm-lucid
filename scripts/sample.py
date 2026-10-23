"""Sample from a trained run: an NFE x guidance sweep of class-conditional grids.

    python scripts/sample.py runs/cifar10_s2-20261024-010203 --ckpt best_fid \
        --nfe 1 2 4 8 --w 1.5 2.0 2.5

The evaluation EMA (θ**) is what gets sampled when the run kept one — it is
the model Algorithm 1 outputs.  Grids go to ``<run>/samples/post/``.
"""

import argparse
import json
import sys
from pathlib import Path

import lucid
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from tvm_lucid import config as config_mod  # noqa: E402
from tvm_lucid.evaluate import save_grid  # noqa: E402
from tvm_lucid.model import DEVICE, build  # noqa: E402
from tvm_lucid.run import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--ckpt", default="last", help="last, best_fid, or epoch_XXXX")
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--w", type=float, nargs="+", default=None, help="guidance weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--live", action="store_true", help="sample the live weights, not θ**")
    args = parser.parse_args()

    cfg = config_mod.from_dict(json.loads((args.run / "config.json").read_text()))
    model = build(cfg.model).eval()
    state = load_checkpoint(args.run / "ckpt" / f"{args.ckpt}.lucid")
    model.load_state_dict(state["model"])
    if "ema" in state and not args.live:
        model.net.load_state_dict(state["ema"])
        which = "ema"
    else:
        which = "live"

    weights = args.w or [model.config.guidance_scale]
    rng = np.random.default_rng(args.seed)
    noise = lucid.tensor(rng.standard_normal((100, 3, 32, 32)).astype(np.float32), device=DEVICE)
    labels = lucid.tensor(np.repeat(np.arange(10), 10), device=DEVICE)
    out_dir = args.run / "samples" / "post"
    out_dir.mkdir(parents=True, exist_ok=True)
    for w in tqdm(weights, desc="guidance"):
        for nfe in tqdm(args.nfe, desc=f"w={w}", leave=False):
            samples = model.generate(
                100, labels=labels, steps=nfe, guidance_scale=w, noise=noise, use_ema=False
            )
            name = f"{args.ckpt}_{which}_w{w:g}_nfe{nfe}.png"
            save_grid(samples.to("cpu").numpy(), out_dir / name)
    print("wrote grids to", out_dir)


if __name__ == "__main__":
    main()

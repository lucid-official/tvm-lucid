"""The released 512² model's learned flow, as heatmaps.

    python scripts/visualize_flow.py --weights ~/.cache/tvm_lucid/im512_w2p5.safetensors \
        --vae path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --out assets

TVM's network is a function of two times, so its flow lives on the triangle
``0 <= s <= t <= 1``.  ``flow_map_plane.png`` shows three fields over it,
measured on ``x_t = (1 - t) x0 + t eps`` (x0: 8-NFE samples, fresh eps):

    speed          RMS of the average velocity u(x_t, t, s) over [s, t]
    straightness   cosine between u(x_t, t, s) and the instantaneous u(x_t, t, t)
    composition    distance between the two-hop jump t -> s -> 0 and the direct jump t -> 0

``decision_time.jpg`` sweeps t from 1 to 0, jumps to 0 from each x_t, and
marks for every pixel the noise level below which it no longer changes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lucid
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_released as ar  # noqa: E402  (also sets the figure style)
from analyze_released import plt  # noqa: E402

from tvm_lucid import pretrained, vae  # noqa: E402

PICK = [1, 6, 9, 15, 0, 3, 12, 13]  # teaser indices: retriever, panda, shuttle, volcano, …
SHOW = [1, 6, 9, 15]
SWEEP = np.linspace(1.0, 0.0, 41)[:-1]  # 40 start times, 1 → 0.025
THRESHOLD = 12.0  # mean |RGB| difference, of 255, that counts as "decided"


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(image, radius, mode="edge")
    rows = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, rows)


def plane(net, x0, eps, labels, w):  # type: ignore[no-untyped-def]
    grid = ar.GRID
    speed = np.full((len(grid), len(grid)), np.nan)
    straight = np.full_like(speed, np.nan)
    compose = np.full_like(speed, np.nan)
    m = len(x0)
    y, wb = ar.T(labels), ar.full(m, w)
    flat = lambda a: a.reshape(m, -1)  # noqa: E731
    rms = lambda a: float(np.sqrt((a**2).mean()))  # noqa: E731
    with lucid.no_grad():
        for ti, t in enumerate(grid):
            if t == 0:
                continue
            xt = (1 - t) * x0 + t * eps
            xt_t = ar.T(xt)
            inst = net(xt_t, ar.full(m, t), ar.full(m, t), y, wb).numpy()
            direct = xt - t * net(xt_t, ar.full(m, t), ar.full(m, 0.0), y, wb).numpy()
            speed[ti, ti] = rms(inst)
            straight[ti, ti] = 1.0
            for si, s in enumerate(grid[:ti]):
                u = net(xt_t, ar.full(m, t), ar.full(m, s), y, wb).numpy()
                speed[ti, si] = rms(u)
                cos = (flat(u) * flat(inst)).sum(1) / (
                    np.linalg.norm(flat(u), axis=1) * np.linalg.norm(flat(inst), axis=1)
                )
                straight[ti, si] = float(cos.mean())
                if s > 0:
                    landing = xt + (s - t) * u
                    hop = net(ar.T(landing), ar.full(m, s), ar.full(m, 0.0), y, wb).numpy()
                    compose[ti, si] = rms(landing - s * hop - direct)
    return speed, straight, compose


def decision_time(net, decoder, x0, eps, label, w, release):  # type: ignore[no-untyped-def]
    """Per pixel: the largest t from which a jump to 0 stays within THRESHOLD of x0's pixel."""
    xts = np.stack([(1 - t) * x0 + t * eps for t in SWEEP]).astype(np.float32)
    jumps = []
    with lucid.no_grad():
        for i in range(0, len(SWEEP), 8):
            ts = SWEEP[i : i + 8]
            m = len(ts)
            u = net(ar.T(xts[i : i + 8]), ar.T(ts.astype(np.float32)), ar.full(m, 0.0),
                    ar.T(np.full(m, label, dtype=np.int64)), ar.full(m, w))  # fmt: skip
            jumps.append(xts[i : i + 8] - ts.reshape(-1, 1, 1, 1).astype(np.float32) * u.numpy())
    pixels = ar.decode(decoder, np.concatenate(jumps), release).astype(np.float32)
    ref = ar.decode(decoder, x0[None], release)[0].astype(np.float32)
    err = np.stack([blur(np.abs(p - ref).mean(axis=-1), 4.0) for p in pixels])  # (len(SWEEP), H, W)
    ok = err <= THRESHOLD
    # decided at SWEEP[k] if every start time from SWEEP[k] down to the smallest is within threshold
    stays = np.flip(np.logical_and.accumulate(np.flip(ok, axis=0), axis=0), axis=0)
    tau = np.where(
        stays.any(axis=0), np.max(np.where(stays, SWEEP[:, None, None], 0.0), axis=0), 0.0
    )
    return ref.astype(np.uint8), tau


def compute(args: argparse.Namespace) -> dict[str, np.ndarray]:
    t0 = time.perf_counter()
    release = pretrained.RELEASES["im512_w2p5"]
    w = release.guidance_scale
    net = lucid.compile(pretrained.load(str(args.weights), release.config(), device=ar.DEVICE))
    decoder = vae.load(str(args.vae), device=ar.DEVICE)
    side = release.latent_size
    labels = np.array(ar.TEASER, dtype=np.int64)
    noise = np.random.default_rng(0).standard_normal((len(labels), 4, side, side))
    noise = noise.astype(np.float32)
    eps = np.random.default_rng(1).standard_normal(noise.shape).astype(np.float32)[PICK]
    with lucid.no_grad():
        x0 = ar.sample(net, noise[PICK], labels[PICK], 8, w)
    print(f"[{time.perf_counter() - t0:.0f}s] x0", flush=True)

    data: dict[str, np.ndarray] = {}
    data["speed"], data["straight"], data["compose"] = plane(net, x0, eps, labels[PICK], w)
    print(f"[{time.perf_counter() - t0:.0f}s] (t, s) plane", flush=True)
    images, taus = [], []
    for i in SHOW:
        j = PICK.index(i)
        image, tau = decision_time(net, decoder, x0[j], eps[j], int(labels[i]), w, release)
        images.append(image)
        taus.append(tau)
    data["images"], data["tau"] = np.stack(images), np.stack(taus)
    print(f"[{time.perf_counter() - t0:.0f}s] decision times", flush=True)
    return data


def numbers(data: dict[str, np.ndarray]) -> dict:
    grid = ar.GRID
    out: dict = {
        name: {
            f"t={grid[a]:.3f},s={grid[b]:.3f}": round(float(data[key][a, b]), 4)
            for a in range(len(grid))
            for b in range(len(grid))
            if not np.isnan(data[key][a, b])
        }
        for name, key in (
            ("speed", "speed"),
            ("straightness", "straight"),
            ("composition", "compose"),
        )
    }
    out["decision_time_mean"] = {
        ar.NAMES[i]: round(float(tau.mean()), 3) for i, tau in zip(SHOW, data["tau"], strict=True)
    }
    return out


def plot(data: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    panels = [
        (data["speed"], "Speed: RMS of u(x_t, t, s)"),
        (data["straight"], "Straightness: cos(u(x_t, t, s), u(x_t, t, t))"),
        (data["compose"], "Composition: |jump t→s→0 − jump t→0|"),
    ]
    for k, (ax, (field, title)) in enumerate(zip(axes, panels, strict=True)):
        ar.triangle(fig, ax, field, title)
        ar.panel_label(ax, "abc"[k])
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "flow_map_plane.png", dpi=150)
    plt.close(fig)

    # Decision time: the sample above, its map below, one colour bar beside the maps.
    n = len(SHOW)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6 + 1.3, 5.6))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.94, bottom=0.01, wspace=0.03, hspace=0.04)
    for c, i in enumerate(SHOW):
        axes[0][c].imshow(data["images"][c])
        axes[0][c].set_title(ar.NAMES[i], fontsize=10, color=ar.INK2)
        im = axes[1][c].imshow(data["tau"][c], cmap=ar.DIVERGING, vmin=0, vmax=1)
        for ax in (axes[0][c], axes[1][c]):
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    # An image keeps its aspect, so its drawn box is smaller than its subplot slot:
    # place the bar after drawing, against the box the last map actually occupies.
    fig.canvas.draw()
    box = axes[1][-1].get_position()
    cax = fig.add_axes((box.x1 + 0.012, box.y0, 0.012, box.height))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("t from which the pixel is decided", color=ar.INK2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=ar.MUTED, labelcolor=ar.INK2)
    tmp = out / "decision_time.tmp.png"
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    Image.open(tmp).convert("RGB").save(out / "decision_time.jpg", quality=90, optimize=True)
    tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--json", type=Path, default=Path("runs/flow.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/flow.npz"))
    parser.add_argument(
        "--replot", action="store_true", help="redraw from --cache, measure nothing"
    )
    args = parser.parse_args()
    if args.replot:
        with np.load(args.cache) as cached:
            data = dict(cached)
    else:
        if args.weights is None or args.vae is None:
            parser.error("--weights and --vae are required unless --replot")
        data = compute(args)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **data)
        args.json.write_text(json.dumps(numbers(data), indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    plot(data, args.out)
    print(f"wrote {args.out}/flow_map_plane.png and {args.out}/decision_time.jpg")


if __name__ == "__main__":
    main()

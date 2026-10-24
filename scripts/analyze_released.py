"""Look inside the released 512² model: attention, the residual stream, the flow map, the terminal velocity.

    python scripts/analyze_released.py --weights ~/.cache/tvm_lucid/im512_w2p5.safetensors \
        --vae path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --out assets

Uses the README teaser's noise and classes (``sample_released.py``, seed 0), so the
figures are about the images shown there.  Writes, into ``--out``:

    analysis_attention.jpg     where a query patch attends, by depth, during the 1-NFE generation
    analysis_pca.jpg           the residual stream's top three principal components, by depth
    analysis_jumps.jpg         one jump to t = 0 from x_t = (1 - t) x0 + t eps, for several t
    analysis_quantitative.png  attention distance, activation scale, terminal-velocity error, NFE

and the numbers behind them to ``--json``.  The internals are read by
``trace``, which is ``TVMNet.forward`` written out with the model's own
modules; it is checked against ``forward`` before anything is recorded.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import lucid
import lucid.nn.functional as F
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvm_lucid import pretrained, vae  # noqa: E402
from tvm_lucid.tvm.layers import modulation, rms_norm, silu, silu_jvp  # noqa: E402

DEVICE = "metal"
# sample_released.py's CLASSES, in order: the teaser's sixteen.
TEASER = [88, 207, 279, 291, 330, 360, 387, 388, 417, 812, 933, 963, 970, 973, 974, 980]
NAMES = {1: "golden retriever", 6: "red panda", 9: "space shuttle", 15: "volcano"}
# (teaser index, (row, col) on the 32x32 token grid, what is there)
QUERIES = [(1, (11, 10), "face"), (1, (26, 24), "grass"), (6, (9, 20), "face"), (15, (3, 8), "sky")]
ATTN_BLOCKS = [1, 7, 13, 19, 25, 27]  # 0-based; shown 1-based
PCA_BLOCKS = [3, 9, 15, 21, 27]
JUMP_IMAGES = [1, 15]
JUMP_TIMES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
RMS_TIMES = [1.0, 0.75, 0.5, 0.25]
NFES = [1, 2, 3, 4, 6, 8, 16]
GRID = [i / 8 for i in range(9)]  # t and s for the terminal-velocity map

# Palette: the dataviz reference instance (light surface; figures are raster).
SURFACE, INK, INK2, MUTED, GRIDC, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#104281"]  # validated --ordinal
# Heatmaps: low blue, high red, through a grey midpoint.  The blue arm is the reference
# ramp; the red arm is stepped to the same OKLab lightness (0.39, 0.53, 0.67, 0.81),
# so neither side reads heavier.  Each arm validates --ordinal (monotone, single hue).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue-grey-red",
    [
        "#104281",
        "#256abf",
        "#5598e7",
        "#9ec5f4",
        "#f0efec",
        "#f1aea8",
        "#dd716a",
        "#b13f3c",
        "#762221",
    ],
)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})  # fmt: skip


def T(array: np.ndarray, **kwargs) -> lucid.Tensor:  # type: ignore[no-untyped-def]
    return lucid.tensor(array, device=DEVICE, **kwargs)


def full(n: int, value: float) -> lucid.Tensor:
    return lucid.full((n,), value, device=DEVICE)


# ── the network, written out ─────────────────────────────────────────────


def token_distances(grid: int) -> lucid.Tensor:
    r, c = np.divmod(np.arange(grid * grid), grid)
    d = np.sqrt((r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2).astype(np.float32)
    return T(d)


def trace(net, x, t, s, labels, w, dist, queries=None, keep_tokens=None):  # type: ignore[no-untyped-def]
    """``net(x, t, s, labels, w)``, recording per block: residual RMS, mean attention
    distance per head, head-averaged attention rows for ``queries`` ({batch: token}),
    and the residual stream for batch entries in ``keep_tokens``."""
    b = int(x.shape[0])
    cond = net.time_embed(t) + net.interval_embed(t - s) + net.guidance_embed(1.0 / w)
    act = silu(cond + net.label_embed(labels))
    tok = net._tokens(x)
    rec = {"rms": [], "dist": [], "attn": {}, "tokens": {}}
    for i, block in enumerate(net.blocks):
        width = int(tok.shape[-1])
        sa, ca, ga, sm, cm, gm = modulation(block.ada_ln[1](act), 6, width, block.modulation_norm)
        attn = block.attn
        q, k, v = attn._heads(attn.qkv(rms_norm(tok) * (1.0 + ca) + sa))
        if attn.qk_norm:
            q, k = rms_norm(q, attn.q_norm.weight), rms_norm(k, attn.k_norm.weight)
        probs = F.softmax((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn.head_dim)), dim=-1)
        tok = tok + ga * attn.proj(attn._merge(probs @ v))
        tok = tok + gm * block.mlp(rms_norm(tok) * (1.0 + cm) + sm)
        # Reading these evaluates the block, so Metal holds one block's activations at a time.
        rec["dist"].append(((probs * dist).sum(dim=-1).mean(dim=-1)).numpy())  # (B, H) tokens
        rec["rms"].append(np.sqrt((tok * tok).reshape(b, -1).mean(dim=-1).numpy()))  # (B,)
        if queries and i in ATTN_BLOCKS:
            rec["attn"][i] = {
                j: probs[j : j + 1, :, q_ : q_ + 1, :].numpy().mean(axis=1)[0, 0]
                for j, q_ in queries.items()
            }
        if keep_tokens and i in PCA_BLOCKS:
            rec["tokens"][i] = {j: tok[j : j + 1].numpy()[0] for j in keep_tokens}
    out = net._scale(net._unpatchify(net.final(tok, act)), w)
    return out, rec


def terminal_velocity(net, x, t, s, labels, w):  # type: ignore[no-untyped-def]
    """``(u, ∂u/∂s)`` as ``TVMNet.jvp_s`` computes them, evaluated block by block."""
    interval, dinterval = net.interval_embed.jvp(t - s, -lucid.ones_like(s))
    cond = net.time_embed(t) + interval + net.guidance_embed(1.0 / w) + net.label_embed(labels)
    act, dact = silu_jvp(cond, dinterval)
    tok = net._tokens(x)
    dtok = lucid.zeros_like(tok)
    for block in net.blocks:
        tok, dtok = block.jvp(tok, dtok, act, dact)
        dtok.reshape(-1)[0].item()
    out, dout = net.final.jvp(tok, dtok, act, dact)
    return (
        net._scale(net._unpatchify(out), w),
        net._scale(net._unpatchify(dout), w),
    )


def sample(net, noise: np.ndarray, labels: np.ndarray, steps: int, w: float, batch: int = 8):  # type: ignore[no-untyped-def]
    out = []
    for i in range(0, len(noise), batch):
        z = pretrained.sample(
            net, T(noise[i : i + batch]), T(labels[i : i + batch]), steps=steps, guidance_scale=w
        )
        out.append(z.numpy())
    return np.concatenate(out)


def decode(decoder, z: np.ndarray, release) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Latents → uint8 images, one per call (eager decoding of several 512² images swaps)."""
    images = []
    for latent in z:
        raw = pretrained.to_vae_latents(T(latent[None]), release)
        images.append(vae.to_pixels(decoder(raw)).numpy().astype(np.uint8)[0])
    return np.stack(images)


# ── figures ──────────────────────────────────────────────────────────────


def overlay(image: np.ndarray, heat: np.ndarray, query: int, gamma: float = 0.7) -> np.ndarray:
    """A head-averaged attention row (32x32) over a dimmed greyscale image, in one hue.

    Scaled to the 99.5th percentile of every patch but the query itself: the query's
    attention to itself would otherwise set the scale and hide where the rest goes.
    The query patch saturates.
    """
    side = image.shape[0]
    flat = heat.reshape(-1)
    heat = np.clip(heat / np.percentile(np.delete(flat, query), 99.5), 0, 1)
    heat = np.asarray(
        Image.fromarray((heat * 255).astype(np.uint8)).resize((side, side), Image.BICUBIC)
    )
    alpha = (heat / 255.0) ** gamma * 0.9
    grey = image.astype(np.float32).mean(axis=-1, keepdims=True) * 0.55
    color = np.array([int(ORANGE[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
    out = grey * (1 - alpha[..., None]) + color * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def image_grid(rows, col_titles, row_titles, path: Path, size: float = 1.9) -> None:  # type: ignore[no-untyped-def]
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * size + 0.9, n_rows * size + 0.45), squeeze=False
    )
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ax = axes[r][c]
            image, marker = cell if isinstance(cell, tuple) else (cell, None)
            ax.imshow(image, interpolation="lanczos" if image.shape[0] > 64 else "nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if marker is not None:  # the query patch: (row, col) on the token grid
                scale = image.shape[0] / 32
                y, x = (marker[0] + 0.5) * scale, (marker[1] + 0.5) * scale
                ax.scatter([x], [y], s=70, c=BLUE, edgecolors="white", linewidths=1.8, zorder=3)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, color=INK2, pad=4)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10, color=INK2, labelpad=6)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.04)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    Image.open(tmp).convert("RGB").save(path, quality=90, optimize=True, progressive=True)
    tmp.unlink()


def pca_rgb(tokens: list[np.ndarray]) -> list[np.ndarray]:
    """Top three principal components of the tokens of several images, fit jointly, as RGB."""
    stacked = np.concatenate([t.astype(np.float64) for t in tokens])
    stacked -= stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(stacked, full_matrices=False)
    proj = stacked @ vt[:3].T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / (hi - lo), 0, 1)
    side = int(math.isqrt(len(tokens[0])))
    maps = np.split(proj, len(tokens))
    return [(m.reshape(side, side, 3) * 255).astype(np.uint8) for m in maps]


def panel_label(ax, letter: str) -> None:  # type: ignore[no-untyped-def]
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)


def colorbar(fig, ax, im, label: str | None = None):  # type: ignore[no-untyped-def]
    """A colour bar exactly as tall as ``ax`` (a plain one is capped by its aspect ratio)."""
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK2)
    if label:
        cbar.set_label(label, color=INK2)
    return cbar


def triangle(fig, ax, field: np.ndarray, title: str, fmt: str = "{:.2f}") -> None:  # type: ignore[no-untyped-def]
    """A field over 0 <= s <= t <= 1 (rows t, columns s on ``GRID``), every cell labelled.

    Blue is the panel's minimum, red its maximum, grey the middle of its range.
    """
    lo, hi = np.nanmin(field), np.nanmax(field)
    half = 0.5 / (len(GRID) - 1)
    im = ax.imshow(np.ma.masked_invalid(field), origin="lower", cmap=DIVERGING, vmin=lo, vmax=hi,
                   aspect="auto", extent=(-half, 1 + half, -half, 1 + half))  # fmt: skip
    for ti, tval in enumerate(GRID):
        for si, sval in enumerate(GRID):
            v = field[ti, si]
            if not np.isnan(v):
                strong = abs((v - lo) / (hi - lo) - 0.5) > 0.3  # deep blue or deep red
                ax.text(sval, tval, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if strong else INK)  # fmt: skip
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("landing time s")
    ax.set_ylabel("start time t")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10)
    colorbar(fig, ax, im)


# ── measurement ──────────────────────────────────────────────────────────


def compute(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    t_start = time.perf_counter()
    release = pretrained.RELEASES["im512_w2p5"]
    w = release.guidance_scale
    net = pretrained.load(str(args.weights), release.config(), device=DEVICE)
    fast = lucid.compile(net)
    decoder = vae.load(str(args.vae), device=DEVICE)
    side, grid = release.latent_size, release.config().grid
    labels = np.array(TEASER, dtype=np.int64)
    noise = (
        np.random.default_rng(0).standard_normal((len(labels), 4, side, side)).astype(np.float32)
    )
    eps = np.random.default_rng(1).standard_normal(noise.shape).astype(np.float32)
    dist = token_distances(grid)
    n = len(labels)
    numbers: dict = {}
    data: dict[str, np.ndarray] = {}

    # The trace is forward, written out; the block-wise JVP is jvp_s.
    with lucid.no_grad():
        x, y = T(noise[:2]), T(labels[:2])
        a, _ = trace(net, x, full(2, 1.0), full(2, 0.0), y, full(2, w), dist)
        gap = float(
            np.abs(a.numpy() - net(x, full(2, 1.0), full(2, 0.0), y, full(2, w)).numpy()).max()
        )
        _, du = terminal_velocity(net, x, full(2, 0.7), full(2, 0.2), y, full(2, w))
        _, du_ref = net.jvp_s(x, full(2, 0.7), full(2, 0.2), y, full(2, w))
        jgap = float(np.abs(du.numpy() - du_ref.numpy()).max() / np.abs(du_ref.numpy()).max())
    print(
        f"trace vs forward: {gap:.1e}; block-wise JVP vs jvp_s: {jgap:.1e} (relative)", flush=True
    )
    assert gap < 1e-4 and jgap < 1e-4
    numbers["check"] = {"trace_vs_forward": gap, "jvp_vs_jvp_s": jgap}

    # 1. The one-step generation, traced: attention, its reach, the residual stream.
    want = [(idx, r * grid + c) for idx, (r, c), _ in QUERIES]
    attn_rows: dict = {}
    tokens: dict = {}
    dist_all, out1 = [], []
    with lucid.no_grad():
        for start in range(0, n, 8):
            idx = range(start, min(start + 8, n))
            xb, yb = T(noise[start : start + 8]), T(labels[start : start + 8])
            m = len(idx)
            here = [(gi, q_) for gi, q_ in want if gi in idx]
            # one query per image per trace: the first trace also keeps the rest of the record
            passes = []
            while here:
                seen, now, later = set(), [], []
                for gi, q_ in here:
                    (later if gi in seen else now).append((gi, q_))
                    seen.add(gi)
                passes.append(now)
                here = later
            passes = passes or [[]]
            for k, now in enumerate(passes):
                local_q = {gi - start: q_ for gi, q_ in now}
                keep = [i - start for i in idx if i in NAMES] if k == 0 else None
                out, rec = trace(
                    net, xb, full(m, 1.0), full(m, 0.0), yb, full(m, w), dist, local_q, keep
                )
                for blk, rows in rec["attn"].items():
                    for j, row in rows.items():
                        attn_rows[(start + j, local_q[j], blk)] = row
                if k == 0:
                    out1.append(out.numpy())
                    dist_all.append(np.stack(rec["dist"], axis=1))  # (B, blocks, heads)
                    for blk, toks in rec["tokens"].items():
                        for j, tok in toks.items():
                            tokens[(start + j, blk)] = tok
    x0_1nfe = noise - np.concatenate(out1)  # x_0 = x_1 + (0 - 1) u
    data["images"] = decode(decoder, x0_1nfe[list(NAMES)], release)
    data["attn_rows"] = np.stack(
        [np.stack([attn_rows[(gi, q_, blk)] for blk in ATTN_BLOCKS]) for gi, q_ in want]
    )
    picks = list(NAMES)
    data["pca"] = np.stack(
        [np.stack(pca_rgb([tokens[(i, blk)] for i in picks])) for blk in PCA_BLOCKS]
    )
    data["head_dist"] = np.concatenate(dist_all).mean(axis=0) * (512 / grid)  # (blocks, heads), px
    data["uniform"] = np.array(float(dist.numpy().mean()) * (512 / grid))
    numbers["attention_distance_px"] = {
        "per_block_mean": data["head_dist"].mean(axis=1).round(1).tolist(),
        "per_block_min": data["head_dist"].min(axis=1).round(1).tolist(),
        "per_block_max": data["head_dist"].max(axis=1).round(1).tolist(),
        "uniform": round(float(data["uniform"]), 1),
    }
    print(f"[{time.perf_counter() - t_start:.0f}s] traced the 1-NFE generation", flush=True)

    # 2. Samples at every step count; the 8-NFE ones stand in for data below.
    with lucid.no_grad():
        by_nfe = {k: sample(fast, noise, labels, k, w) for k in NFES}
    x0, ref = by_nfe[8], by_nfe[16]
    data["nfe_dist"] = np.stack(
        [np.sqrt(((by_nfe[k] - ref) ** 2).reshape(n, -1).mean(axis=1)) for k in NFES]
    )
    numbers["distance_to_16_nfe"] = {
        str(k): [round(float(v.mean()), 4), round(float(v.std()), 4)]
        for k, v in zip(NFES, data["nfe_dist"], strict=True)
    }
    print(f"[{time.perf_counter() - t_start:.0f}s] samples at {NFES} NFE", flush=True)

    # 3. Residual-stream scale by depth at several t (x_t as in training: fresh eps).
    rms = []
    with lucid.no_grad():
        for tv in RMS_TIMES:
            xt = (1 - tv) * x0 + tv * eps
            per = []
            for start in range(0, n, 8):
                m = min(8, n - start)
                _, rec = trace(
                    net, T(xt[start : start + 8]), full(m, tv), full(m, 0.0),
                    T(labels[start : start + 8]), full(m, w), dist,
                )  # fmt: skip
                per.append(np.stack(rec["rms"], axis=1))
            rms.append(np.concatenate(per).mean(axis=0))
    data["rms"] = np.stack(rms)
    numbers["residual_rms"] = {
        str(t): r.round(3).tolist() for t, r in zip(RMS_TIMES, rms, strict=True)
    }

    # 4. Terminal-velocity error over (t, s): d/ds f against the field where the jump lands.
    tv_err = np.full((len(GRID), len(GRID)), np.nan)
    pick = [1, 6, 9, 15, 0, 3, 12, 13]
    with lucid.no_grad():
        for ti, tval in enumerate(GRID):
            for si, sval in enumerate(GRID):
                if not 0 < tval or sval >= tval:
                    continue
                m = len(pick)
                xt = T((1 - tval) * x0[pick] + tval * eps[pick])
                yb, wb = T(labels[pick]), full(m, w)
                tt, ss = full(m, tval), full(m, sval)
                u, du = terminal_velocity(net, xt, tt, ss, yb, wb)
                velocity = u + (sval - tval) * du  # d/ds f, with f = (s - t) u
                target = net(xt + (sval - tval) * u, ss, ss, yb, wb)
                diff = (velocity - target).numpy().reshape(m, -1)
                norm = target.numpy().reshape(m, -1)
                tv_err[ti, si] = float(
                    np.mean(np.linalg.norm(diff, axis=1) / np.linalg.norm(norm, axis=1))
                )
    data["tv_err"] = tv_err
    numbers["terminal_velocity_error"] = {
        f"t={GRID[ti]:.3f},s={GRID[si]:.3f}": round(float(tv_err[ti, si]), 4)
        for ti in range(len(GRID))
        for si in range(len(GRID))
        if not np.isnan(tv_err[ti, si])
    }
    print(f"[{time.perf_counter() - t_start:.0f}s] terminal-velocity map", flush=True)

    # 5. One jump to 0 from several t.
    jump_rows, jump_delta = [], {}
    with lucid.no_grad():
        for idx in JUMP_IMAGES:
            xts, jumps = [], []
            for tval in JUMP_TIMES:
                xt = (1 - tval) * x0[idx] + tval * eps[idx]
                if tval == 0:
                    jump = xt
                else:
                    lab = T(labels[idx : idx + 1])
                    u = net(T(xt[None]), full(1, tval), full(1, 0.0), lab, full(1, w))
                    jump = xt - tval * u.numpy()[0]
                jump_delta[f"{NAMES[idx]},t={tval:g}"] = round(
                    float(np.sqrt(((jump - x0[idx]) ** 2).mean())), 4
                )
                xts.append(xt)
                jumps.append(jump)
            jump_rows.append(decode(decoder, np.stack(xts), release))
            jump_rows.append(decode(decoder, np.stack(jumps), release))
    data["jumps"] = np.stack(jump_rows)
    numbers["jump_rms_to_x0"] = jump_delta
    print(f"[{time.perf_counter() - t_start:.0f}s] measured", flush=True)
    return data, numbers


# ── plotting ─────────────────────────────────────────────────────────────


def plot(data: dict[str, np.ndarray], out: Path) -> None:
    grid = 32
    images = dict(zip(NAMES, data["images"], strict=True))

    rows, row_titles = [], []
    for k, (idx, (r, c), what) in enumerate(QUERIES):
        img = images[idx]
        q_ = r * grid + c
        row = [(img, (r, c))]
        row += [
            overlay(img, data["attn_rows"][k, b].reshape(grid, grid), q_)
            for b in range(len(ATTN_BLOCKS))
        ]
        rows.append(row)
        row_titles.append(f"{NAMES[idx]} · {what}")
    image_grid(
        rows,
        ["query patch"] + [f"block {b + 1}" for b in ATTN_BLOCKS],
        row_titles,
        out / "analysis_attention.jpg",
    )

    picks = list(NAMES)
    rows = [
        [images[i]] + [data["pca"][b, j] for b in range(len(PCA_BLOCKS))]
        for j, i in enumerate(picks)
    ]
    image_grid(
        rows,
        ["1-NFE sample"] + [f"block {b + 1}" for b in PCA_BLOCKS],
        [NAMES[i] for i in picks],
        out / "analysis_pca.jpg",
    )

    rows, row_titles = [], []
    for k, idx in enumerate(JUMP_IMAGES):
        rows += [list(data["jumps"][2 * k]), list(data["jumps"][2 * k + 1])]
        row_titles += [f"{NAMES[idx]} · $x_t$", f"{NAMES[idx]} · one jump"]
    image_grid(rows, [f"t = {t:g}" for t in JUMP_TIMES], row_titles, out / "analysis_jumps.jpg")

    head_dist, uniform, rms, tv_err = (
        data["head_dist"],
        float(data["uniform"]),
        data["rms"],
        data["tv_err"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    blocks = np.arange(1, head_dist.shape[0] + 1)

    ax = axes[0][0]
    for h in range(head_dist.shape[1]):
        ax.scatter(blocks, head_dist[:, h], s=10, color=BLUE, alpha=0.35, linewidths=0)
    ax.plot(blocks, head_dist.mean(axis=1), color=BLUE, lw=2)
    ax.axhline(uniform, color=MUTED, lw=1, ls="--")
    ax.text(28.6, uniform, "uniform\nattention", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0.5, 31.5)
    ax.set_xlabel("block")
    ax.set_ylabel("mean attention distance (px)")
    ax.set_title("Attention reach by depth, 1-NFE generation (dots: 18 heads)", loc="left")
    panel_label(ax, "a")

    ax = axes[0][1]
    for color, tv, line in zip(ORDINAL, RMS_TIMES, rms, strict=True):
        ax.plot(blocks, line, color=color, lw=2, label=f"t = {tv:g}")
    ax.set_xlim(0.5, 28.5)
    ax.set_xlabel("block")
    ax.set_ylabel("residual-stream RMS")
    ax.set_title("Activation scale by depth, one jump t → 0", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    panel_label(ax, "b")

    ax = axes[1][0]
    triangle(fig, ax, tv_err, "Terminal-velocity error ‖d/ds f − u(x_s, s)‖ / ‖u(x_s, s)‖")
    panel_label(ax, "c")

    ax = axes[1][1]
    nfes = NFES[:-1]
    mean, std = data["nfe_dist"][:-1].mean(axis=1), data["nfe_dist"][:-1].std(axis=1)
    ax.fill_between(nfes, mean - std, mean + std, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(nfes, mean, color=BLUE, lw=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5)  # fmt: skip
    for k, v in zip(nfes, mean, strict=True):
        ax.text(k, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(nfes, [str(k) for k in nfes])
    ax.set_xlabel("NFE")
    ax.set_ylabel("latent RMS distance to the 16-NFE sample")
    ax.set_ylim(0, None)
    ax.set_title("Distance to the 16-NFE sample, 16 classes (±1 sd)", loc="left")
    panel_label(ax, "d")

    fig.tight_layout(w_pad=3, h_pad=2.5)
    fig.savefig(out / "analysis_quantitative.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--json", type=Path, default=Path("runs/analysis.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/analysis.npz"))
    parser.add_argument(
        "--replot", action="store_true", help="redraw from --cache, measure nothing"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with np.load(args.cache) as cached:
            data = dict(cached)
    else:
        if args.weights is None or args.vae is None:
            parser.error("--weights and --vae are required unless --replot")
        data, numbers = compute(args)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **data)
        args.json.write_text(json.dumps(numbers, indent=1))
    plot(data, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()

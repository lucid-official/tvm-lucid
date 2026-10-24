"""Sample a released ImageNet TVM checkpoint on Apple Silicon, and time each backend.

    python -I scripts/convert_released.py im512_w2p5.pkl ~/.cache/tvm_lucid/im512_w2p5.safetensors
    python scripts/sample_released.py --weights ~/.cache/tvm_lucid/im512_w2p5.safetensors \
        --vae path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
        --backend eager compile coreml --nfe 1 2 4

Backends — every one runs the same network on the same noise:

    eager    TVMNet and the VAE decoder as written, float32 on Metal
    compile  lucid.compile of both: one MPSGraph executable per input signature
    coreml   lucid.coreml packages, float16 on the GPU, batch 1; exported on
             first use into ~/.cache/tvm_lucid/coreml and loaded from there after

The generator runs ``--batch`` images per call (Core ML: 1); the VAE decodes
one image per call on every backend, because eager decoding of several 512px
images at once outgrows a 36 GB Mac's memory.  Grids (one class per column)
and ``timings.json`` go to ``--out``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lucid
import lucid.coreml as cml
import lucid.metal
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from tvm_lucid import pretrained, vae  # noqa: E402

# Sixteen ImageNet classes that are easy to judge by eye.
CLASSES = {
    88: "macaw", 207: "golden retriever", 279: "arctic fox", 291: "lion",
    330: "wood rabbit", 360: "otter", 387: "red panda", 388: "giant panda",
    417: "balloon", 812: "space shuttle", 933: "cheeseburger", 963: "pizza",
    970: "alp", 973: "coral reef", 974: "geyser", 980: "volcano",
}  # fmt: skip
CACHE = Path.home() / ".cache" / "tvm_lucid" / "coreml"


def save_grid(pixels: np.ndarray, path: Path, cols: int) -> None:
    """``(N, H, W, 3)`` uint8 → one PNG, ``cols`` images per row."""
    n, h, w, _ = pixels.shape
    rows = -(-n // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, image in enumerate(pixels):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = image
    Image.fromarray(canvas).save(path)


class Backend:
    """A generator ``net(x, t, s, labels, w)`` and a decoder ``decode(z)``, on one device."""

    device = "metal"
    batch: int | None = None  # None: --batch

    def net(self, *args: lucid.Tensor) -> lucid.Tensor:
        raise NotImplementedError

    def decode(self, z: lucid.Tensor) -> lucid.Tensor:
        raise NotImplementedError


class Eager(Backend):
    def __init__(self, args: argparse.Namespace, release: pretrained.Release) -> None:
        self._net = pretrained.load(str(args.weights), release.config(), device="metal")
        self._vae = vae.load(str(args.vae), device="metal")

    def net(self, *args: lucid.Tensor) -> lucid.Tensor:
        return self._net(*args)

    def decode(self, z: lucid.Tensor) -> lucid.Tensor:
        return self._vae(z)


class Compiled(Eager):
    def __init__(self, args: argparse.Namespace, release: pretrained.Release) -> None:
        super().__init__(args, release)
        self._net = lucid.compile(self._net)
        self._vae = lucid.compile(self._vae)


class CoreML(Backend):
    device = "cpu"  # Core ML reads host memory
    batch = 1

    def __init__(self, args: argparse.Namespace, release: pretrained.Release) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        side = release.latent_size
        # Not the Neural Engine: it ran these networks 2.5-5x slower, with 10-25x the error.
        units = cml.ComputeUnits.CPU_AND_GPU
        net_path = CACHE / f"{release.name}_b1_fp16.mlpackage"
        vae_path = CACHE / f"sd-vae-ft-ema_{side * 8}px_b1_fp16.mlpackage"
        example_z = lucid.zeros(1, 4, side, side)
        if not net_path.exists():
            net = pretrained.load(str(args.weights), release.config(), device="cpu")
            example = (
                example_z,
                lucid.ones(1),
                lucid.zeros(1),
                lucid.zeros(1, dtype=lucid.int64),
                lucid.full((1,), release.guidance_scale),
            )
            print(f"exporting {net_path.name} (once) …", flush=True)
            cml.export(
                net, example, str(net_path), precision=cml.Precision.FLOAT16, compute_units=units
            ).close()
            del net
        if not vae_path.exists():
            decoder = vae.load(str(args.vae), device="cpu")
            print(f"exporting {vae_path.name} (once) …", flush=True)
            cml.export(
                decoder,
                example_z,
                str(vae_path),
                precision=cml.Precision.FLOAT16,
                compute_units=units,
            ).close()
            del decoder
        self._net = cml.load(str(net_path), compute_units=units)
        self._vae = cml.load(str(vae_path), compute_units=units)

    def net(self, *args: lucid.Tensor) -> lucid.Tensor:
        return self._net.predict(args)  # type: ignore[return-value]

    def decode(self, z: lucid.Tensor) -> lucid.Tensor:
        return self._vae.predict(z)  # type: ignore[return-value]


BACKENDS = {"eager": Eager, "compile": Compiled, "coreml": CoreML}


def run(
    backend: Backend,
    release: pretrained.Release,
    noise: np.ndarray,
    labels: np.ndarray,
    steps: int,
    guidance: float,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Latents and pixels for every image, with generator and decoder seconds."""
    batch = backend.batch or batch
    latents, pixels = [], []
    gen_s = dec_s = 0.0
    for start in range(0, len(noise), batch):
        x = lucid.tensor(noise[start : start + batch], device=backend.device)
        y = lucid.tensor(labels[start : start + batch], device=backend.device)
        t0 = time.perf_counter()
        z = pretrained.sample(backend.net, x, y, steps=steps, guidance_scale=guidance)
        z_host = z.to("cpu").numpy()
        gen_s += time.perf_counter() - t0
        latents.append(z_host)
        raw = pretrained.to_vae_latents(z, release)
        for i in range(int(raw.shape[0])):
            t0 = time.perf_counter()
            image = vae.to_pixels(backend.decode(raw[i : i + 1]))
            pixels.append(image.to("cpu").numpy().astype(np.uint8))
            dec_s += time.perf_counter() - t0
    return np.concatenate(latents), np.concatenate(pixels), gen_s, dec_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help="converted .safetensors")
    parser.add_argument("--vae", type=Path, required=True, help="sd-vae-ft-ema .safetensors")
    parser.add_argument("--release", default="im512_w2p5", choices=sorted(pretrained.RELEASES))
    parser.add_argument("--backend", nargs="+", default=["compile"], choices=list(BACKENDS))
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--w", type=float, default=None, help="guidance (default: the release's)")
    parser.add_argument("--classes", type=int, nargs="+", default=list(CLASSES))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/released"))
    args = parser.parse_args()

    release = pretrained.RELEASES[args.release]
    guidance = args.w if args.w is not None else release.guidance_scale
    labels = np.array(args.classes, dtype=np.int64)
    side = release.latent_size
    noise = np.random.default_rng(args.seed).standard_normal((len(labels), 4, side, side))
    noise = noise.astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    cols = min(len(labels), 4)

    timings: dict[str, dict] = {}
    reference: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for name in args.backend:
        t0 = time.perf_counter()
        backend = BACKENDS[name](args, release)
        setup = time.perf_counter() - t0
        # One untimed batch per step count, at the timed batch's shape: compile
        # traces once per signature, Core ML specialises on first use.
        warm = min(backend.batch or args.batch, len(labels))
        for steps in args.nfe:
            run(backend, release, noise[:warm], labels[:warm], steps, guidance, args.batch)
        timings[name] = {"setup_s": round(setup, 2)}
        for steps in args.nfe:
            z, pixels, gen_s, dec_s = run(
                backend, release, noise, labels, steps, guidance, args.batch
            )
            path = args.out / f"{release.name}_{name}_nfe{steps}.png"
            save_grid(pixels, path, cols)
            n = len(labels)
            row = {
                "generator_ms_per_image": round(gen_s / n * 1e3, 1),
                "vae_ms_per_image": round(dec_s / n * 1e3, 1),
                "images_per_s": round(n / (gen_s + dec_s), 2),
            }
            if steps in reference:  # agreement with the first backend
                z_ref, px_ref = reference[steps]
                mse = float(np.mean((pixels.astype(np.float64) - px_ref) ** 2))
                row["latent_max_abs_diff_vs_" + args.backend[0]] = float(np.abs(z - z_ref).max())
                row["psnr_db_vs_" + args.backend[0]] = (
                    round(10 * np.log10(255.0**2 / mse), 1) if mse else float("inf")
                )
            else:
                reference[steps] = (z, pixels.astype(np.float64))
            timings[name][f"nfe{steps}"] = row
            print(f"{name:8s} nfe={steps}: {json.dumps(row)}  → {path}", flush=True)
        del backend
        if hasattr(lucid.metal, "empty_cache"):
            lucid.metal.empty_cache()

    meta = {
        "release": release.name,
        "guidance": guidance,
        "classes": {int(c): CLASSES.get(int(c), "") for c in labels},
        "seed": args.seed,
        "batch": args.batch,
        "lucid": lucid.__version__,
    }
    (args.out / "timings.json").write_text(json.dumps({**meta, "backends": timings}, indent=2))
    print("wrote", args.out / "timings.json")


if __name__ == "__main__":
    main()

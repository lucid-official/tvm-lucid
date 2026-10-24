"""A one-page web app for the released ImageNet models: pick a model, a class and a step count.

    LUCID_COREML_CACHE_LIMIT=8G python webui/server.py \
        --vae path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors \
        --host "$(tailscale ip -4)" --port 8512

Then open ``http://<host>:8512`` from any device that can reach that address.
Bind it to a private address (a Tailscale one, or 127.0.0.1 behind an SSH
tunnel): there is no authentication.

Every release whose converted weights are in ``--weights-dir``
(``<name>.safetensors``) can be chosen.  One model is loaded at a time, by
``scripts/sample_released.py``'s backends, and one worker thread makes
every Lucid / Core ML call: switching models and generating images are jobs
in the same queue, so a model is never swapped under a running generation,
and the old one is closed before the new one loads.  A generation reports
its progress jump by jump; the page polls ``/job/<id>``.

Images are written to ``--out`` and served from there, with ``?download=1``
for a download.  With Core ML, keep ``LUCID_COREML_CACHE_LIMIT`` above the
compiled size of every model (1.3 GB each) or a switch recompiles one.
"""

import argparse
import gc
import json
import queue
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import lucid
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sample_released  # noqa: E402

from tvm_lucid import pretrained, vae  # noqa: E402

CLASSES: list[str] = json.loads((HERE / "imagenet_classes.json").read_text())
ORDER = ("im512_w2p5", "im256_w2", "im256_w1p75")  # as the page lists them
assert set(ORDER) == set(pretrained.RELEASES)
MAX_NFE = 16
KEEP_JOBS = 200
_NAME = re.compile(r"^[\w.-]+\.png$")


@dataclass
class Job:
    """One generation, as the page sees it while it runs."""

    id: str
    model: str
    label: int
    nfe: int
    w: float
    seed: int
    phase: str = "queued"  # queued, loading, network, vae, done, error
    step: int = 0
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def view(self, position: int | None) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in ("id", "model", "phase", "step", "nfe")}
        out.update(result=self.result, error=self.error, queue_position=position)
        return out


class Worker:
    """The loaded model and the one thread that touches it."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.models = {name: args.weights_dir / f"{name}.safetensors" for name in ORDER}
        self.lock = threading.Lock()  # guards everything below that the HTTP threads read
        self.loaded: str | None = None
        self.loading: str | None = None
        self.load_error: dict[str, str] = {}
        self.timing: dict[str, dict[str, float]] = {}  # model → measured ms per jump / VAE
        self.jobs: dict[str, Job] = {}
        self.pending: list[str] = []  # generation ids in queue order
        self.queue: queue.Queue = queue.Queue()
        self.backend: sample_released.Backend | None = None
        threading.Thread(target=self._run, name="worker", daemon=True).start()

    # ── called from HTTP threads ─────────────────────────────────────────

    def available(self, name: str) -> bool:
        return name in self.models and self.models[name].is_file()

    def request_load(self, name: str) -> None:
        self.queue.put(("load", name))

    def submit(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self.pending.append(job.id)
            for stale in list(self.jobs)[:-KEEP_JOBS]:
                if self.jobs[stale].phase in ("done", "error"):
                    del self.jobs[stale]
        self.queue.put(("generate", job.id))

    def view(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            position = self.pending.index(job_id) if job_id in self.pending else None
            return job.view(position)

    def status(self) -> dict[str, Any]:
        with self.lock:
            models = []
            for name in ORDER:
                release = pretrained.RELEASES[name]
                models.append({
                    "name": name, "image_size": release.image_size, "default_w": release.guidance_scale,
                    "fid": {str(k): v for k, v in release.fid.items()}, "available": self.available(name),
                    "error": self.load_error.get(name), "timing": self.timing.get(name),
                })  # fmt: skip
            return {
                "models": models,
                "loaded": self.loaded,
                "loading": self.loading,
                "backend": self.args.backend,
                "queued": len(self.pending),
                "max_nfe": MAX_NFE,
            }

    # ── the worker thread ────────────────────────────────────────────────

    def _run(self) -> None:
        if self.available(self.args.model):
            self._switch(self.args.model)
        while True:
            kind, payload = self.queue.get()
            if kind == "load":
                self._switch(payload)
                continue
            job = self.jobs.get(payload)
            if job is None:
                continue
            try:
                if not self._switch(job.model, job):
                    raise RuntimeError(self.load_error.get(job.model, f"cannot load {job.model}"))
                result = self._generate(job)
                with self.lock:
                    job.result, job.phase = result, "done"
            except Exception as error:  # reported on the job, the worker carries on
                with self.lock:
                    job.error, job.phase = f"{type(error).__name__}: {error}", "error"
            finally:
                with self.lock:
                    if job.id in self.pending:
                        self.pending.remove(job.id)

    def _unload(self) -> None:
        if self.backend is not None:
            self.backend.close()
        self.backend = None
        gc.collect()
        lucid.metal.empty_cache()

    def _switch(self, name: str, job: Job | None = None) -> bool:
        """Make ``name`` the loaded model, closing the old one first. False if it cannot load."""
        if self.loaded == name and self.backend is not None:
            return True
        if not self.available(name):
            with self.lock:
                self.load_error[name] = "weights not converted"
            return False
        with self.lock:
            self.loading = name
            if job is not None:
                job.phase = "loading"
        try:
            with self.lock:
                self.loaded = None
            self._unload()
            release = pretrained.RELEASES[name]
            args = argparse.Namespace(weights=self.models[name], vae=self.args.vae, batch=1)
            t0 = time.perf_counter()
            backend = sample_released.BACKENDS[self.args.backend](args, release)
            self.backend = backend
            # Warm up (compile traces, Core ML specialises on the first call), then time a
            # settled run, so the page's estimate is right from the first generation.
            for _ in range(2):
                self._render(release, 0, 2, release.guidance_scale, 0, name)
            print(
                f"loaded {name} on {self.args.backend} in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            with self.lock:
                self.loaded = name
                self.load_error.pop(name, None)
            return True
        except Exception as error:
            self._unload()
            with self.lock:
                self.load_error[name] = f"{type(error).__name__}: {error}"
            print(f"failed to load {name}: {error!r}", flush=True)
            return False
        finally:
            with self.lock:
                self.loading = None

    def _render(self, release, label, nfe, w, seed, model, job: Job | None = None):  # type: ignore[no-untyped-def]
        """Generate one image; returns uint8 pixels and the two timings in ms."""
        backend = self.backend
        assert backend is not None
        side = release.latent_size
        noise = np.random.default_rng(seed).standard_normal((1, 4, side, side)).astype(np.float32)
        x = lucid.tensor(noise, device=backend.device)
        y = lucid.tensor(np.array([label], dtype=np.int64), device=backend.device)

        def on_step(k: int, x: lucid.Tensor) -> None:
            x.reshape(-1)[0].item()  # Metal is lazy: make the progress real
            if job is not None:
                with self.lock:
                    job.step = k

        if job is not None:
            with self.lock:
                job.phase, job.step = "network", 0
        t0 = time.perf_counter()
        z = pretrained.sample(backend.net, x, y, steps=nfe, guidance_scale=w, on_step=on_step)
        t1 = time.perf_counter()
        if job is not None:
            with self.lock:
                job.phase = "vae"
        image = vae.to_pixels(backend.decode(pretrained.to_vae_latents(z, release)))
        pixels = image.to("cpu").numpy().astype(np.uint8)[0]
        t2 = time.perf_counter()
        step_ms, vae_ms = (t1 - t0) * 1e3 / nfe, (t2 - t1) * 1e3
        with self.lock:
            self.timing[model] = {"ms_per_step": round(step_ms, 1), "vae_ms": round(vae_ms, 1)}
        return pixels, step_ms * nfe, vae_ms

    def _generate(self, job: Job) -> dict[str, Any]:
        release = pretrained.RELEASES[job.model]
        pixels, net_ms, vae_ms = self._render(
            release, job.label, job.nfe, job.w, job.seed, job.model, job
        )
        slug = re.sub(r"[^a-z0-9]+", "-", CLASSES[job.label].lower()).strip("-")
        name = (
            f"{time.strftime('%Y%m%d-%H%M%S')}_{job.model}_{job.label:03d}-{slug}"
            f"_nfe{job.nfe}_w{job.w:g}_s{job.seed}.png"
        )
        Image.fromarray(pixels).save(self.args.out / name)
        return {
            "model": job.model,
            "image_size": release.image_size,
            "label": job.label,
            "class_name": CLASSES[job.label],
            "nfe": job.nfe,
            "w": job.w,
            "seed": job.seed,
            "network_ms": round(net_ms),
            "vae_ms": round(vae_ms),
            "name": name,
            "url": f"/img/{name}",
        }


class Handler(BaseHTTPRequestHandler):
    worker: Worker
    out: Path

    def log_message(self, format: str, *args: object) -> None:  # quiet: the page polls
        if "/job/" not in self.path and self.path != "/status":
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: object) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 1024:
            return None
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/classes.json":
            self._json(200, CLASSES)
        elif path == "/status":
            self._json(200, self.worker.status())
        elif path.startswith("/job/"):
            view = self.worker.view(path[len("/job/") :])
            if view is None:
                self._json(404, {"error": "no such job"})
            else:
                self._json(200, view)
        elif path.startswith("/img/") and _NAME.match(name := path[len("/img/") :]):
            file = self.out / name
            if not file.is_file():
                self._json(404, {"error": "no such image"})
                return
            extra = {}
            if "download=1" in query.split("&"):
                extra["Content-Disposition"] = f'attachment; filename="{name}"'
            self._send(200, file.read_bytes(), "image/png", extra)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        if body is None:
            self._json(400, {"error": "bad request"})
            return
        if self.path == "/model":
            name = str(body.get("model", ""))
            if not self.worker.available(name):
                self._json(400, {"error": f"{name!r} is not a converted release"})
                return
            self.worker.request_load(name)
            self._json(202, {"loading": name})
        elif self.path == "/generate":
            self._generate(body)
        else:
            self._json(404, {"error": "not found"})

    def _generate(self, body: dict) -> None:
        try:
            model = str(body.get("model") or self.worker.loaded or self.worker.args.model)
            release = pretrained.RELEASES[model]
            label = int(body["label"])
            nfe = int(body.get("nfe", 1))
            w = float(body.get("w", release.guidance_scale))
            seed = body.get("seed")
            seed = secrets.randbelow(2**31) if seed in (None, "") else int(seed)
        except ValueError, KeyError, TypeError:
            self._json(400, {"error": "model, label, nfe, w and seed are required as numbers"})
            return
        if not self.worker.available(model):
            self._json(400, {"error": f"{model} is not converted"})
            return
        if not (0 <= label < len(CLASSES) and 1 <= nfe <= MAX_NFE and 1.0 <= w <= 8.0
                and 0 <= seed < 2**32):  # fmt: skip
            self._json(400, {"error": f"label 0-999, nfe 1-{MAX_NFE}, w 1-8, seed 0-2^32"})
            return
        job = Job(secrets.token_hex(8), model, label, nfe, w, seed)
        self.worker.submit(job)
        self._json(202, {"job": job.id})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path.home() / ".cache" / "tvm_lucid",
        help="holds <release>.safetensors from scripts/convert_released.py",
    )
    parser.add_argument("--vae", type=Path, required=True, help="sd-vae-ft-ema .safetensors")
    parser.add_argument("--model", default="im512_w2p5", choices=sorted(pretrained.RELEASES))
    parser.add_argument("--backend", default="coreml", choices=list(sample_released.BACKENDS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8512)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "webui")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    Handler.worker = Worker(args)
    Handler.out = args.out
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

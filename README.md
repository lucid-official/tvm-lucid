<h1 align="center">TVM on Lucid</h1>

<p align="center">
  <b>One-step 512×512 ImageNet generation, natively on Apple Silicon.</b><br>
  A torch-free re-implementation of <a href="https://arxiv.org/abs/2511.19797">Terminal Velocity Matching</a>
  on <a href="https://github.com/lucid-official/lucid">lucid-dl</a>. The released 678M checkpoints run unchanged on
  Metal, verified against the reference to 1.4e-5, and the method trains from scratch on one Mac.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.19797"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2511.19797-b31b1b.svg"></a>
  <a href="https://github.com/lumalabs/tvm"><img alt="Upstream" src="https://img.shields.io/badge/upstream-lumalabs%2Ftvm-black.svg"></a>
  <a href="https://huggingface.co/lumaai/tvm"><img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-lumaai%2Ftvm-orange.svg"></a>
  <a href="https://pypi.org/project/lucid-dl/3.15.5/"><img alt="lucid" src="https://img.shields.io/badge/lucid--dl-3.15.5-4c8bf5.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="100%">
</p>

<p align="center">
  <sub>The released ImageNet-512 checkpoint (<code>im512_w2p5</code>, <i>w</i> = 2.5) at <b>one network evaluation
  per image</b>, 512×512, on a Mac Studio M4 Max. The classes were picked by hand; the samples were not: seed 0,
  first draw for each class. Generated with <code>lucid.compile</code>: 112 ms per image for the network, 216 ms for
  the VAE.</sub>
</p>

---

## Highlights

- **The authors' weights, unchanged.** All three released checkpoints map one-to-one onto this port's network. The
  512² one runs here with outputs within **1.4e-5** of the reference implementation.
- **No pickle is ever executed.** The release ships EDM-style pickles that `exec` embedded source code when loaded.
  A restricted reader turns them into SafeTensors without running anything and without torch.
- **One step, 512², 0.28 s end to end.** Exported through `lucid.coreml`, the 678M network generates an image in
  102 ms and the VAE decodes it in 181 ms. `lucid.compile` gives 1.63× on the network at fp32 with no visible
  change: 84 dB PSNR against eager.
- **Trains from scratch, too.** The full method, including the Jacobian-vector product that the loss
  back-propagates through, is written on Lucid's public API and trains on Metal end to end.

## News

- **2026-10-25** — The released ImageNet checkpoints run on Lucid, with a pickle-free converter, an SD-VAE decoder,
  and Metal, `lucid.compile` and Core ML backends. See [Results](#results) and [Speed](#speed-on-an-m4-max).
- **2026-10-24** — First from-scratch CIFAR-10 run (S/2, 33M) started on a Mac Studio M4 Max. See
  [Training from scratch](#training-from-scratch-cifar-10).

## What this is

Terminal Velocity Matching (Zhou, Parger, Haque & Song, ICLR 2026) generalises flow matching from the
instantaneous velocity to the **flow map**: one network models the jump between *any* two times, so one evaluation
can go from noise to data. TVM supervises that jump at its **terminal** time. The derivative of the jump with respect
to where it lands must equal the velocity field at the landing point. The paper proves that this bounds the
2-Wasserstein distance when the network is Lipschitz, and it makes the minimal changes to DiT that give it that
property.

This repository contains:

- **The method, written as a library user would write it.** The network, the objective, the time sampler, both EMAs
  and the sampler use only `lucid`, `lucid.nn`, `lucid.nn.functional`, `lucid.linalg`, `lucid.special` and
  `lucid.optim`. Nothing comes from Lucid's private modules or its model zoo, and
  [`tests/test_public_api.py`](tests/test_public_api.py) enforces this.
- **The released ImageNet models.** A converter for the upstream checkpoints, the SD-VAE decoder they need, and a
  sampler that runs on eager Metal, `lucid.compile` or Core ML.
- **A from-scratch training run and its harness.** Class-conditional CIFAR-10 in pixel space with an S/2 backbone on
  one Mac. The harness adds deterministic resumable data, per-epoch sample grids from fixed noise, flow-map
  consistency, a per-time velocity probe, FID/IS, and JSONL logs of the training dynamics.

## Results

### ImageNet 512×512 from the released checkpoint

The authors released three checkpoints ([`lumaai/tvm`](https://huggingface.co/lumaai/tvm)). All three are
DiT-XL/2 with 18 heads (678,068,496 parameters) over the latents of `stabilityai/sd-vae-ft-ema`:

| checkpoint | resolution | latents | guidance *w* | `do_scale_param` | FID-50k, as reported upstream |
|---|---|---|---:|:---:|---|
| `im256_w2` | 256² | 32×32×4 | 2.0 | ✓ | 3.29 at 1 NFE |
| `im256_w1p75` | 256² | 32×32×4 | 1.75 | ✓ | 1.99 at 4 NFE |
| **`im512_w2p5`** | **512²** | **64×64×4** | **2.5** | ✗ | **4.32 at 1 NFE** |

512² is the largest resolution released, and it is the checkpoint run here. The FIDs are the authors'. This
repository samples the models and does not re-measure them.

### One network, any number of steps

The same network serves every step count: `steps` even jumps from *t* = 1 to *t* = 0 (Figure 3). More steps keep
the composition and sharpen what the single jump blurred, such as the otter's face, the shuttle's markings and the
reef fish.

<p align="center">
  <img src="assets/nfe.jpg" width="100%"><br>
  <sub>Rows: 1, 2 and 4 NFE, from the same noise. Otter, space shuttle, coral reef, pizza, giant panda, golden
  retriever.</sub>
</p>

### At full resolution

<p align="center">
  <img src="assets/full_resolution.jpg" width="80%"><br>
  <sub>Native 512×512 crops, not resized. Left: 1 NFE. Right: 4 NFE. Red panda, coral reef.</sub>
</p>

## Speed on an M4 Max

Measured by [`scripts/sample_released.py`](scripts/sample_released.py) on a Mac Studio M4 Max (32-core GPU,
36 GB), with a working checkout of Lucid `main` at `7e47e33` and macOS 27.0. The run takes 16 images at 512². The network runs 8 images
per call (Core ML: 1) and the VAE one image per call. Every backend gets the same noise, and an untimed batch of
the timed shape runs first.

| backend | precision | network, 1 NFE | VAE decode | end to end, 1 NFE | images/s at 1 · 2 · 4 NFE | agreement with eager |
|---|---|---:|---:|---:|---|---|
| Metal, eager | fp32 | 183 ms | 237 ms | 420 ms | 2.38 · 1.65 · 1.03 | — |
| `lucid.compile` | fp32 | **112 ms** (1.63×) | 216 ms | 328 ms | 3.05 · 2.27 · 1.50 | 84 dB PSNR |
| `lucid.coreml`, GPU | fp16 | **102 ms** (1.79×) | **181 ms** | **284 ms** | **3.53 · 2.59 · 1.70** | 56 dB PSNR (45 dB at 4 NFE) |

<p align="center">
  <img src="assets/backends.jpg" width="100%"><br>
  <sub>The same noise and class at 1 NFE. Eager, <code>lucid.compile</code>, Core ML fp16, then
  |compile − eager| and |Core ML − eager|, both ×20. Compile differs by at most 1 level of 255; fp16 by at most
  14, along edges.</sub>
</p>

**Batch size.** `lucid.compile` caches one MPSGraph executable per input signature. Its first call traces in about
1.6 s.

| network batch | eager | `lucid.compile` | speedup |
|---:|---:|---:|---:|
| 1 | 175 ms | 121 ms | 1.45× |
| 4 | 182 ms / image | 111 ms / image | 1.63× |
| 8 | 184 ms / image | 112 ms / image | 1.65× |

**The Neural Engine is the wrong target for these networks.** Core ML places most of their operations on the ANE
when asked to, and it is slower and less exact there:

| fp16, batch 1 | GPU | Neural Engine | `ALL` (Core ML chooses) |
|---|---|---|---|
| two DiT-XL blocks, 1024 tokens | **7.9 ms**, 8.7e-4 rel. | 19.8 ms (84% of ops on ANE), 8.6e-3 rel. | 12.0 ms, chose the GPU |
| VAE decoder, 512² | **181 ms**, 6.3e-3 abs. | 922 ms (94% of ops on ANE), 0.16 abs. | — |

What this settles:

- **Core ML on the GPU is the fastest path at batch 1**, and fp16 costs nothing visible: 56 dB at 1 NFE. Its error
  compounds over steps, down to 45 dB at 4 NFE.
- **`lucid.compile` is the fast fp32 path**, with no change in output and no export step. Its gain grows with batch
  size and levels off from batch 4.
- **At 512², the VAE costs more than the generator.** On every backend one decode (181–237 ms) takes longer than one
  network evaluation (102–183 ms). A one-step model at this resolution is decoder-bound.

Two costs to plan for:

- **Exporting the generator** takes 5.4 s and peaks at 18 GiB of host memory at batch 1, for a 1.3 GB package. At
  batch 4 it ran a 36 GB Mac out of memory and swap.
- **Eager VAE decoding** of more than one 512² image per call pushes Metal into swap: 714 ms per image at batch 4,
  against 218 ms compiled. The sampler decodes one image per call.

## Numerical parity

Against the reference implementations, given the same weights:

| check | against | result |
|---|---|---:|
| `TVMNet`, **`im512_w2p5` weights**, fp32 on Metal | upstream `DiT_XL_2_18h`, fp32 | 1.4e-5 rel. (5.9e-5 abs.) |
| `TVMNet`, random DiT-S/2 weights, both parameterisations, CPU and Metal | upstream `DiT` | ≤ 2.6e-6 rel. |
| SD-VAE decoder, `sd-vae-ft-ema` | diffusers `AutoencoderKL`, fp32 | 5.7e-5 max, 2.5e-6 mean |
| converter | a pickle written by upstream's own `persistence` module | bit-identical, 164 / 164 tensors |
| positional table this port rebuilds | the table stored in the release | < 1e-5, checked on every load |

The network checks include conditional and null labels and per-sample guidance weights. Upstream ran inference
under bf16 autocast. This port runs in fp32 (eager, compile) or fp16 (Core ML).

The method itself is checked against independent derivations:

| check | against | result |
|---|---|---:|
| `jvp_s` primal | `forward`, same weights | 8.9e-15 |
| $\partial_s u_\theta$ | central finite difference in $s$ | 5.4e-9 (outputs O(1)) |
| $\partial_s u_\theta$ | JVP by double backward | 2.7e-15 |
| $\nabla_\theta \langle \partial_s u_\theta, R\rangle$, all 42 tensors | double-backward reference | 1.5e-14 rel. |
| Eq. 15 loss / gradients | independent re-derivation, same random draws | 0.0 / 2.8e-15 rel. |
| `qk_norm`, `modulation_norm`, scaled param. off | finite difference in $s$ | ≤ 1.9e-8 |

These figures are float64 and were measured during development by running this exact source on CPU. lucid-dl
ships macOS wheels only, so that run used a stand-in that serves Lucid's API names from PyTorch, and the reference
values came from PyTorch's own double backward.

[`tests/test_tvm_method.py`](tests/test_tvm_method.py) re-checks the same properties on Lucid itself, on CPU and
Metal, with float32 tolerances. It also checks:

- the untrained field is exactly zero (adaLN-Zero);
- linear layers start at unit spectral norm;
- the block is invariant to the scale of its modulation (Eq. 16), and is not invariant without it;
- the scaled parameterisation multiplies by $w$;
- every time sampler keeps $s \le t$;
- the terminal term vanishes at $s = t$;
- both EMAs follow Algorithm 1 and stay out of `state_dict()`;
- the sampler takes Figure 3's jumps.

[`tests/test_pretrained.py`](tests/test_pretrained.py) pins the release's names onto this port's. On a Lucid
checkout that ships `lucid.models.generative.tvm`, `TestZooParity` loads the same weights into both implementations
and compares outputs, losses and gradients. The PyPI release skips it.

## Getting started

### Install

You need an Apple Silicon Mac on macOS 26 or later, and Python 3.14.

**Lucid's `main` (recommended until the next release).** The PyPI release 3.15.5 handles Metal views in a Python
layer that `main` replaced with C++ strided views, and the same training step was about 4× slower on it. With a
built Lucid checkout, use that checkout's interpreter and run from this directory, with nothing installed:

```bash
git clone https://github.com/lucid-official/tvm-lucid && cd tvm-lucid
LUCID_PY=~/path/to/lucid/.venv/bin/python             # the checkout's own venv
uv pip install --python $LUCID_PY tqdm pillow matplotlib
$LUCID_PY -c "import lucid; print(lucid.__file__)"   # must point into the checkout
$LUCID_PY -m pytest -q                               # also runs the parity tests against Lucid's zoo TVM
```

Every command below then starts with `$LUCID_PY` in place of `python`. Each training run records the checkout's
commit, and whether it had local changes, in `env.json`. Do not rebuild or sync the checkout while a run is going:
modules imported later in the run would come from the new code.

**PyPI `lucid-dl` 3.15.5.**

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -e ".[test]"
python -m pytest -q
```

3.15.5 declares `mlx>=0.31` with no upper bound, but it is linked against mlx 0.32.3. mlx 0.32.4 changed
`mlx::core::array::set_data`, so on a fresh install Lucid fails to import. `pyproject.toml` therefore pins
`mlx==0.32.3`. 3.15.5 has every API the released-weights path uses (SafeTensors, `lucid.compile`,
`lucid.coreml`), but the timings above were measured on `main` only.

### Run the released weights

```bash
# the checkpoint, and the VAE it was trained with (2.7 GB + 0.3 GB)
curl -L -o im512_w2p5.pkl https://huggingface.co/lumaai/tvm/resolve/main/im512_w2p5.pkl
curl -L -o sd-vae-ft-ema.safetensors \
    https://huggingface.co/stabilityai/sd-vae-ft-ema/resolve/main/diffusion_pytorch_model.safetensors

# pickle → SafeTensors without unpickling: no torch, nothing in the file is executed
python -I scripts/convert_released.py im512_w2p5.pkl im512_w2p5.safetensors

# 16 classes at 1, 2 and 4 NFE on every backend; grids and timings.json in runs/released/
python scripts/sample_released.py --weights im512_w2p5.safetensors --vae sd-vae-ft-ema.safetensors \
    --backend eager compile coreml --nfe 1 2 4
```

`--classes` takes any ImageNet indices, `--w` another guidance weight, and `--release` one of the 256² checkpoints.
The Core ML packages are exported on first use into `~/.cache/tvm_lucid/coreml` and loaded from there afterwards.

From Python:

```python
import lucid
from tvm_lucid import pretrained, vae

release = pretrained.RELEASES["im512_w2p5"]
net = lucid.compile(pretrained.load("im512_w2p5.safetensors", release.config()))
decoder = vae.load("sd-vae-ft-ema.safetensors")

noise = lucid.randn(4, 4, 64, 64, device="metal")
labels = lucid.tensor([88, 207, 387, 980], device="metal")  # macaw, golden retriever, red panda, volcano
z = pretrained.sample(net, noise, labels, steps=1, guidance_scale=release.guidance_scale)
z = pretrained.to_vae_latents(z, release)
images = [vae.to_pixels(decoder(z[i : i + 1])) for i in range(4)]  # (1, 512, 512, 3) each, 0-255
```

### Train from scratch

```bash
# end-to-end check of the whole pipeline (minutes): train, log, checkpoint, grids, probe, FID
python -m tvm_lucid.train --config configs/debug.toml

# throughput and peak memory, then the epochs that fit the budget
python -m tvm_lucid.train --config configs/cifar10_s2.toml --benchmark

# the run (survives closing the ssh session)
nohup caffeinate -dimsu python -m tvm_lucid.train --config configs/cifar10_s2.toml > train.out 2>&1 &

# resume where it stopped; any field can be overridden
python -m tvm_lucid.train --resume runs/cifar10_s2-<timestamp> --set train.hours=24

# plots and NFE × guidance grids, also while training
python scripts/plot_dynamics.py runs/cifar10_s2-<timestamp>
python scripts/sample.py runs/cifar10_s2-<timestamp> --ckpt best_fid --nfe 1 2 4 8 --w 1.5 2 2.5
```

On first use, CIFAR-10 is downloaded to `~/datasets`. The default source is a byte-identical Hugging Face mirror,
and the official Toronto server, which is often throttled to about 100 KiB/s, is the fallback. Both are checked
against the official md5. The first evaluation also caches the FID reference statistics there.

## Training from scratch: CIFAR-10

> **Status.** The first full run is paused after 27 epochs (21,822 steps, about 23 h of its 84 h budget) and
> resumes with `--resume`. Its samples, FID curves and timings will be added when it finishes. Nothing in this
> section is a trained-model result.

The training hyperparameters are the released ImageNet-256 recipe
([`configs/im256_w2.yaml`](https://github.com/lumalabs/tvm/blob/main/configs/im256_w2.yaml)) wherever it carries
over. Only the data, the backbone and the batch differ:

| | this run | released (ImageNet-256, `im256_w2`) |
|---|---|---|
| data | CIFAR-10 train split, 50k images, pixels in [−1, 1], no flips | SD-VAE latents, no flips |
| backbone | S/2: width 384, depth 12, 6 heads, patch 2 → 256 tokens, **32.97M** | XL/2: 1152, 28, 18 heads → 256 tokens, 678M |
| optimiser | AdamW, lr 1e-4, β = (0.9, 0.95), eps 1e-8, no weight decay, no clipping | same |
| batch | **64** | **2048** |
| guidance | constant $w = 2$, label dropout 0.1, scaled parameterisation | same |
| times | gap*: gap ~ logit-N(−0.8, 1), $s$ ~ logit-N(−0.4, 1) truncated | same |
| EMAs | $\theta^*$ 0.99, $\theta^{**}$ 0.9999 | same |
| init | spectral, time embeddings included | same |
| budget | 84 h on a Mac Studio M4 Max (36 GB): about 82k steps at 3.7 s, about 100 epochs | — |

**Why the batch is not 2048.** Accumulating 2048 here would take about 2 minutes per step, or about 2.5k steps in
the budget. The evaluation EMA alone averages over about 10k steps ($1/(1-0.9999)$), so it would never leave its
initial weights. At 64 the same budget gives about 82k steps. The recipe's per-step values carry over unchanged,
but its guarantees were measured at its own batch, data and model.

Everything in the table is one line of [`configs/cifar10_s2.toml`](configs/cifar10_s2.toml), where each value names
the released field it copies. Every `TVMConfig` field can be set under `[model]`.

### Live dashboard

`runs/<run>/dashboard.png` is redrawn at the start of the run and after every epoch (and every
`eval.dashboard_every_steps` steps, if set). Each redraw replaces the file in one step, so a viewer never reads half
an image: Preview normally picks up each new version, and `dashboard.html` shows the same image in any browser,
reloading it every 30 seconds.

| | left | right |
|---|---|---|
| top | **loss**: total, terminal and flow matching, as running loss per iteration (lines) and epoch means (markers), step and epoch axes | **samples**: FID by NFE, and the held-out flow-matching error of $\theta^{**}$ |
| middle | **optimisation**: global gradient norm and update ratio $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ | **consistency**: flow-map gap (K-step vs N-step) and sample std against the data's |
| bottom | the 1- and 2-step samplers of $\theta^{**}$ from the same noise $x_1$: the noise, the 1-NFE sample, the 2-NFE midpoint and sample, one class per column (both columns) | |

The FID and consistency curves likewise show only `eval.dashboard_nfe` (1 and 2 NFE). The full set of dynamics
plots (per-block norms, EMA lag, activation RMS, error by time) is `scripts/plot_dynamics.py`.

### What gets measured

| measurement | when | what it tells you |
|---|---|---|
| `loss`, `terminal`, `fm` (and `*_mean`) | every step | Eq. 15 and its two terms; `*_mean` averages every step since the last row |
| `train_loss`, `train_terminal`, `train_fm` | every epoch | the same, averaged over the epoch |
| gradient / parameter norms per block | `dynamics_every` | a block collapsing or exploding before the loss shows it |
| update ratio $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ | `dynamics_every` | the step size Adam actually applies |
| EMA drift of $\theta^*$, $\theta^{**}$ | `dynamics_every` | how far the target and evaluation networks lag |
| time-embedding / raw adaLN RMS | `dynamics_every` | the quantity in Figure 4 (DiT without Lipschitz control blows up) |
| 10×10 sample grids, fixed noise | every epoch | the same cell, epoch after epoch |
| `flowmap_gap_KvN` | every epoch | MSE between K-step and N-step samples from the same noise |
| `probe_fm_{live,ema}` | every epoch | held-out flow-matching error in 10 bands of $s$: where in time the field is still wrong |
| `fid_nfeK`, `is_nfeK` | `fid_every_epochs` | 5k class-balanced samples against the training split |

```
runs/<name>-<timestamp>/
  dashboard.png dashboard.html           the live dashboard, rewritten in place
  config.json   env.json                 resolved config; host, Lucid version and commit, start time
  metrics.jsonl                          kind=step and kind=dynamics rows
  epoch.jsonl   events.jsonl             per-epoch evaluation; start / stop / resume / skips / best FID
  samples/      ckpt/     plots/         grids; last, best_fid, epoch_XXXX; plot_dynamics.py output
```

## How it works

With $x_t = (1-t)\,x + t\,\epsilon$ and the network's **average velocity** $u_\theta(x_t, t, s) = w\,F_\theta(x_t, t, s, c, w)$
over $[s, t]$, the flow map is

$$f_\theta(x_t, t, s) = (s - t)\,u_\theta(x_t, t, s), \qquad x_s \approx x_t + f_\theta(x_t, t, s).$$

Its **terminal velocity** is $\tfrac{d}{ds} f_\theta = u_\theta + (s - t)\,\partial_s u_\theta$. Eq. 15 trains it
against the field where the jump lands. A guided flow-matching term pins down that field:

$$\mathcal{L} = \tfrac{1}{w^2}\Bigl\lVert \tfrac{d}{ds} f_\theta - u_{\theta^*}\bigl(\mathrm{sg}[x_t + f_\theta],\, s\bigr)\Bigr\rVert^2 \mathbb{1}_{t\ne s}
\;+\; \tfrac{1}{w^2}\bigl\lVert u_\theta(x_{s'}, s') - [\,w(\epsilon - x) + (1 - w)\,u_{\theta^*}(x_{s'}, s', \varnothing)\,]\bigr\rVert^2 .$$

The targets come from $\theta^*$, an EMA of the weights, under a stop-gradient. The gradient flows **through**
$\partial_s u_\theta$. Sampling takes `steps` even jumps from $t = 1$ to $t = 0$ (Figure 3), and the same network
serves every step count. Under `do_scale_param` (the 256² releases and the CIFAR-10 run) the network predicts $F$
and the velocity is $wF$. `im512_w2p5` was trained without it and predicts the velocity directly.

```
x_t ──▶ patchify ──▶ + 2-D sin-cos table ─────────────────────────┐
                                                                  ▼
t ─────▶ sinusoid ─▶ MLP ──┐                         N × semi-Lipschitz DiT block
t − s ─▶ sinusoid ─▶ MLP ──┤   (the tangent along s       (parameter-free RMSNorm, QK-norm,
1/w ───▶ sinusoid ─▶ MLP ──┼─▶ Σ ─▶ cond ─────────────▶  RMS-normalised adaLN shift/scale/gate)
class ─▶ label table ──────┘    enters here)                      │
                                                                  ▼
                                       modulated RMSNorm ─▶ linear ─▶ unpatchify ─▶ × w ─▶ u_θ
```

| component | paper | port |
|---|---|---|
| semi-Lipschitz DiT block: RMSNorm, QK-norm, normalised modulation, spectral init | §4, App. B.2, Table 6 | [`tvm_lucid/tvm/layers.py`](tvm_lucid/tvm/layers.py) |
| two-time network conditioned on $t$, $t - s$, $1/w$, class | §4, App. F.1 | [`tvm_lucid/tvm/network.py`](tvm_lucid/tvm/network.py) |
| derivative along $s$ that stays on the autograd graph | Eq. 15, §5 | `jvp` methods in `layers.py`, `TVMNet.jvp_s` |
| objective, gap* time sampler, CFG target, $1/w^2$ weight | Eq. 15, §4, Table 4 | [`tvm_lucid/tvm/method.py`](tvm_lucid/tvm/method.py) |
| target EMA $\theta^*$ and evaluation EMA $\theta^{**}$ | Algorithm 1, Table 4b | `TVM.update_ema` |
| one- and few-step sampler | Figure 3 | `TVM.generate`, `pretrained.sample` |
| every paper default and ablation switch | Tables 4, 6 | [`tvm_lucid/tvm/config.py`](tvm_lucid/tvm/config.py) |
| released checkpoints: configs, names, latent statistics | — | [`tvm_lucid/pretrained.py`](tvm_lucid/pretrained.py) |
| SD-VAE decoder (`sd-vae-ft-ema`) | — | [`tvm_lucid/vae.py`](tvm_lucid/vae.py) |

Some details follow the released code where the paper leaves them open:

- the scale keeps DiT's `1 +`;
- every norm uses ε = 1e-6;
- the terminal term is masked with `isclose(s, t)`, not exact equality;
- squared norms are averaged over elements;
- truncated logit-normal draws use the inverse CDF.

At the paper's defaults the network has **678,068,496** parameters, which is the paper's 678M for XL/2 and exactly
the parameter count of the released checkpoints.

### Loading a release without running it

The upstream checkpoints are EDM-style pickles. `torch_utils.persistence` stores each network class's *source code*
in the file and `exec`s it on load, so `pickle.load` runs whatever the file says. The release wraps the network in
`torch.compile`'s `OptimizedModule` as well. [`scripts/convert_released.py`](scripts/convert_released.py) reads it
with an unpickler whose every global resolves to an inert stub, except the few that rebuild tensors. Those slice
the legacy storage format out of the bytes with numpy. The result for `im512_w2p5`:

- 356 tensors and 679,248,144 elements: the parameters plus the 1024×1152 frozen positional table;
- written as SafeTensors under the upstream names in seconds, with nothing executed and no torch imported.

[`tvm_lucid/pretrained.py`](tvm_lucid/pretrained.py) loads the file with `lucid.serialization.load_safetensors`,
renames each tensor onto `TVMNet`, and rebuilds the positional table instead of copying it. The mapping is total
in both directions.

### The JVP, without `lucid.func.jvp`

TVM's prediction is itself a Jacobian-vector product, and the loss back-propagates through it. The paper writes a
fused attention kernel for exactly that. In lucid-dl 3.15.5, neither JVP fits:

- `lucid.func.jvp` returns its tangent **detached**. That is right for MeanFlow, whose JVP sits under a
  stop-gradient, but not for TVM.
- `lucid.autograd.functional.jvp` is a finite difference.

So every layer here carries its own forward-mode rule: a `jvp` method that maps `(input, tangent)` to
`(output, tangent)`. Each rule is built from the same differentiable ops as `forward`:

| layer | rule |
|---|---|
| linear | $\dot y = W\dot x$ |
| SiLU | $\dot y = \dot x\,\sigma(x)\,(1 + x(1-\sigma(x)))$ |
| GELU (tanh) | $\dot y = \dot x\,[\tfrac12(1+\tanh u) + \tfrac12 x(1-\tanh^2 u)\,u']$ |
| RMSNorm | $\dot y = r^{-1}(\dot x - y\,\overline{y\dot x})$, $\;r = \sqrt{\overline{x^2}+\epsilon}$ |
| attention | $\dot A = A\odot(\dot S - \textstyle\sum_j A\odot\dot S)$, $\;\dot o = \dot A V + A\dot V$ |

One sweep through the network carries the activations and their $s$-tangents side by side, and `backward()`
differentiates the tangent like any other activation. Compared with a JVP by double backward, this costs a
first-order backward instead of a third-order graph.

`forward` is written with the same primitives as the primal half of `jvp`: attention as matmul and softmax, with
GELU, RMSNorm and SiLU as elementwise formulas, rather than `F.scaled_dot_product_attention`, `F.gelu` and
`F.rms_norm`. Under autograd on Metal those three were the slow part of a training step. The flow-matching forward
that used them cost 0.12 s per image, against 0.026 s per forward-equivalent for the written-out path (S/2,
batch 64). It also makes `forward` and the primal of `jvp` the same computation. The attention matrix is
materialised either way. That is small at 256 tokens, and 72 MiB per image and block at the 1024 tokens of the 512²
release.

## Repository layout

```
tvm_lucid/
  tvm/            the method: config.py, layers.py, network.py, method.py
  pretrained.py   the released ImageNet checkpoints: configs, name mapping, sampler, latent statistics
  vae.py          SD-VAE decoder for the released latents
  model.py        TVM sized for CIFAR-10
  data.py         download, md5 check, (seed, epoch, batch)-deterministic batches
  train.py        Algorithm 1 loop, tqdm, JSONL logging, checkpoints, budget, non-finite guard
  evaluate.py     grids, flow-map gap, per-time probe, FID / IS
  dynamics.py     grouped norms, update ratio, EMA drift, activation probes
  inception.py    pretrained Inception-v3 features (lucid.models.inception_v3_cls)
  stats.py        FID / IS arithmetic (numpy)
  run.py          run directories, environment capture, atomic checkpoints
scripts/
  convert_released.py   release pickle → SafeTensors, without unpickling
  sample_released.py    released weights: grids and timings on eager / compile / Core ML
  sample.py             NFE × guidance grids from a training run
  plot_dynamics.py      training-dynamics plots
  profile_step.py       per-phase timing of a training step
configs/          cifar10_s2.toml, debug.toml
assets/           README figures, cut from sample_released.py grids
```

## Limitations

- **The released models are sampled, not evaluated.** No FID-50k is computed here. The table quotes the authors'
  numbers, which were measured under bf16 autocast on their hardware.
- **Only `im512_w2p5` has been run.** The 256² checkpoints are configured (`--release im256_w2`, `im256_w1p75`)
  and go through the same converter and name mapping, but neither has been downloaded and sampled here.
- **Core ML at batch 1.** Exporting the 678M network peaks at 18 GiB of host memory at batch 1 and exhausted a
  36 GB Mac at batch 4. Larger batches need more memory than this machine has.
- **FID on CIFAR-10 is not the paper's FID.** It uses Lucid's ImageNet `inception_v3_cls`, not the TF-Inception
  graph that published numbers use. It compares runs of this repository with each other, not with tables in papers.
- **Training runs eagerly.** The step reads $\theta^*$ between the live forward and the loss, while the compiled
  training step in `lucid.compile` traces a single `model(x)`. Whether it can capture both without baking
  $\theta^*$ into the cached executable has not been measured yet, so training does not use `lucid.compile`.
- **Resuming is deterministic but not bit-identical.** Metal's generator state cannot be checkpointed, so a resumed
  run reseeds its own draws from `(seed, step)`. The data order is exact either way.

## Citation

```bibtex
@article{zhou2025terminal,
  title={Terminal Velocity Matching},
  author={Zhou, Linqi and Parger, Mathias and Haque, Ayaan and Song, Jiaming},
  journal={arXiv preprint arXiv:2511.19797},
  year={2025}
}
```

## Acknowledgements

- [Terminal Velocity Matching](https://github.com/lumalabs/tvm), for the method, the released code that settles
  the details the paper leaves open, and the [released checkpoints](https://huggingface.co/lumaai/tvm).
- [DiT](https://github.com/facebookresearch/DiT), for the backbone it modifies.
- [`stabilityai/sd-vae-ft-ema`](https://huggingface.co/stabilityai/sd-vae-ft-ema), for the autoencoder the
  ImageNet models generate in.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009), for the data.
- [lucid-dl](https://github.com/lucid-official/lucid), for the Apple Silicon runtime.

No model weights or datasets are redistributed here. Both checkpoints and CIFAR-10 are downloaded by the user. The
images in [`assets/`](assets/) were generated by the released `im512_w2p5` checkpoint with this code.

## License

Code is MIT.

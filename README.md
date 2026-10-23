<h1 align="center">TVM on Lucid</h1>

<p align="center">
  <b>One- and few-step generation, trained from scratch on Apple Silicon.</b><br>
  A torch-free re-implementation of <a href="https://arxiv.org/abs/2511.19797">Terminal Velocity Matching</a>
  on <a href="https://github.com/lucid-official/lucid">lucid-dl</a>. It uses Lucid's public API only and trains on Metal from end to end.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.19797"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2511.19797-b31b1b.svg"></a>
  <a href="https://github.com/lumalabs/tvm"><img alt="Upstream" src="https://img.shields.io/badge/upstream-lumalabs%2Ftvm-black.svg"></a>
  <a href="https://pypi.org/project/lucid-dl/3.15.5/"><img alt="lucid" src="https://img.shields.io/badge/lucid--dl-3.15.5-4c8bf5.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

> **Status.** The implementation is complete and verified (see [Verification](#verification)). The first full
> CIFAR-10 run is in progress on a Mac Studio M4 Max. Its samples, FID curves and timings will be added here when
> it finishes. Nothing below is a trained-model result.

---

## What this is

Terminal Velocity Matching (Zhou, Parger, Haque & Song, ICLR 2026) generalises flow matching from the
instantaneous velocity to the **flow map**: one network models the jump between *any* two times, so one evaluation
can go from noise to data. TVM supervises that jump at its **terminal** time. The derivative of the jump with respect
to where it lands must equal the velocity field at the landing point. The paper proves that this bounds the
2-Wasserstein distance when the network is Lipschitz, and it makes the minimal changes to DiT that give it that
property.

This repository contains three things:

- **The method, written as a library user would write it.** The network, the objective, the time sampler, both EMAs
  and the sampler are implemented here using only `lucid`, `lucid.nn`, `lucid.nn.functional`, `lucid.linalg`,
  `lucid.special` and `lucid.optim`. Nothing is imported from Lucid's private modules or from its model zoo, and
  [`tests/test_public_api.py`](tests/test_public_api.py) enforces this.
- **A from-scratch training run.** Class-conditional CIFAR-10 in pixel space, with an S/2-sized backbone, on one Mac.
- **A research harness around it.** Deterministic resumable data and per-epoch sample grids from fixed noise.
  Flow-map consistency, a per-time velocity probe, FID/IS, and JSONL logs of the training dynamics.

## The method in four lines

With $x_t = (1-t)\,x + t\,\epsilon$ and the network's **average velocity** $u_\theta(x_t, t, s) = w\,F_\theta(x_t, t, s, c, w)$
over $[s, t]$, the flow map is

$$f_\theta(x_t, t, s) = (s - t)\,u_\theta(x_t, t, s), \qquad x_s \approx x_t + f_\theta(x_t, t, s).$$

Its **terminal velocity** is $\tfrac{d}{ds} f_\theta = u_\theta + (s - t)\,\partial_s u_\theta$. Eq. 15 trains it
against the field where the jump lands. A guided flow-matching term pins down that field:

$$\mathcal{L} = \tfrac{1}{w^2}\Bigl\lVert \tfrac{d}{ds} f_\theta - u_{\theta^*}\bigl(\mathrm{sg}[x_t + f_\theta],\, s\bigr)\Bigr\rVert^2 \mathbb{1}_{t\ne s}
\;+\; \tfrac{1}{w^2}\bigl\lVert u_\theta(x_{s'}, s') - [\,w(\epsilon - x) + (1 - w)\,u_{\theta^*}(x_{s'}, s', \varnothing)\,]\bigr\rVert^2 .$$

The targets come from $\theta^*$, an EMA of the weights, under a stop-gradient. The gradient flows **through**
$\partial_s u_\theta$. Sampling takes `steps` even jumps from $t = 1$ to $t = 0$ (Figure 3), and the same network
serves every step count.

## How it works

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
| one- and few-step sampler | Figure 3 | `TVM.generate` |
| every paper default and ablation switch | Tables 4, 6 | [`tvm_lucid/tvm/config.py`](tvm_lucid/tvm/config.py) |

Some details follow the released code where the paper leaves them open:

- the scale keeps DiT's `1 +`;
- every norm uses ε = 1e-6;
- the terminal term is masked with `isclose(s, t)`, not exact equality;
- squared norms are averaged over elements;
- truncated logit-normal draws use the inverse CDF.

At the paper's defaults the network has **678,068,496** parameters, which is the paper's 678M for XL/2.

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
first-order backward instead of a third-order graph. Only the tangent stream materialises the attention matrix: with
256 tokens per CIFAR image it is small. Every other attention call stays on `F.scaled_dot_product_attention`.

## Verification

| check | against | result |
|---|---|---:|
| `jvp_s` primal | `forward`, same weights | 8.9e-15 |
| $\partial_s u_\theta$ | central finite difference in $s$ | 5.4e-9 (outputs O(1)) |
| $\partial_s u_\theta$ | JVP by double backward | 2.7e-15 |
| $\nabla_\theta \langle \partial_s u_\theta, R\rangle$, all 42 tensors | double-backward reference | 1.5e-14 rel. |
| Eq. 15 loss / gradients | independent re-derivation, same random draws | 0.0 / 2.8e-15 rel. |
| `qk_norm`, `modulation_norm`, scaled param. off | finite difference in $s$ | ≤ 1.9e-8 |

These figures are float64 and were measured during development by running this exact source on CPU. lucid-dl ships
macOS wheels only, so the run used a stand-in that serves Lucid's API names from PyTorch. Reference values came from
PyTorch's own double backward.

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

On a Lucid checkout that ships `lucid.models.generative.tvm`, `TestZooParity` loads the same weights into both
implementations and compares outputs, losses and gradients. The PyPI release skips it.

## The CIFAR-10 run

| | |
|---|---|
| data | CIFAR-10 train split, 50k images, pixel space 32×32 in [−1, 1], horizontal flips |
| backbone | S/2: width 384, depth 12, 6 heads, patch 2 → 256 tokens, **32.97M** parameters |
| guidance | constant $w = 2$ (the paper's best one-step weight), class dropout 0.1 |
| times | gap*: gap ~ logit-N(−0.8, 1), $s$ ~ logit-N(−0.4, 1) truncated, separate flow-matching time |
| EMAs | $\theta^*$ 0.99 (Table 4b), $\theta^{**}$ 0.999 (paper: 0.9999 over a much longer run) |
| optimiser | AdamW, lr 1e-4, β = (0.9, 0.95), batch 64, no clipping, no weight decay |
| budget | 84 h of wall clock on a Mac Studio M4 Max (36 GB), stopping cleanly with a checkpoint |

Everything in the table is one line of [`configs/cifar10_s2.toml`](configs/cifar10_s2.toml), and every `TVMConfig`
field can be set under `[model]`. Compared with the paper's ImageNet setting, this run is in pixel space rather than
SD-VAE latent space, uses a smaller backbone, uses a shorter evaluation EMA, and adds flips.

### What gets measured

| measurement | when | what it tells you |
|---|---|---|
| `loss`, `terminal`, `fm` | every step | Eq. 15 and its two terms; `terminal` is 0 on $s = t$ samples |
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
  config.json   env.json                 resolved config; host, lucid-dl version, start time
  metrics.jsonl                          kind=step and kind=dynamics rows
  epoch.jsonl   events.jsonl             per-epoch evaluation; start / stop / resume / skips / best FID
  samples/      ckpt/     plots/         grids; last, best_fid, epoch_XXXX; plot_dynamics.py output
```

## Install

You need an Apple Silicon Mac on macOS 26 or later, and Python 3.14. The code uses Lucid's public API only, so it
runs on either Lucid below.

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

Every command below then starts with `$LUCID_PY` in place of `python`. Each run records the checkout's commit, and
whether it had local changes, in `env.json`. Do not rebuild or sync the checkout while a run is going: modules
that are imported later in the run would come from the new code.

**PyPI `lucid-dl` 3.15.5.**

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -e ".[test]"
python -m pytest -q
```

3.15.5 declares `mlx>=0.31` with no upper bound, but it is linked against mlx 0.32.3. mlx 0.32.4 changed
`mlx::core::array::set_data`, so on a fresh install Lucid fails to import. `pyproject.toml` therefore pins
`mlx==0.32.3`.

## Quick start

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

```
tvm_lucid/
  tvm/            the method: config.py, layers.py, network.py, method.py
  model.py        TVM sized for CIFAR-10
  data.py         download, md5 check, (seed, epoch, batch)-deterministic batches
  train.py        Algorithm 1 loop, tqdm, JSONL logging, checkpoints, budget, non-finite guard
  evaluate.py     grids, flow-map gap, per-time probe, FID / IS
  dynamics.py     grouped norms, update ratio, EMA drift, activation probes
  inception.py    pretrained Inception-v3 features (lucid.models.inception_v3_cls)
  stats.py        FID / IS arithmetic (numpy)
  run.py          run directories, environment capture, atomic checkpoints
scripts/          plot_dynamics.py, sample.py
configs/          cifar10_s2.toml, debug.toml
```

## Limitations

- **FID here is not the paper's FID.** It uses Lucid's ImageNet `inception_v3_cls`, not the TF-Inception graph that
  published numbers use. It compares runs of this repository with each other, not with tables in papers.
- **Training runs eagerly.** The step reads $\theta^*$ between the live forward and the loss, while the compiled
  training step in `lucid.compile` traces a single `model(x)`. Whether it can capture both without baking
  $\theta^*$ into the cached executable has not been measured yet, so `lucid.compile` is not used.
- **Resuming is deterministic but not bit-identical.** Metal's generator state cannot be checkpointed, so a resumed
  run reseeds its own draws from `(seed, step)`. The data order is exact either way.
- **No pretrained weights.** The released ImageNet checkpoints are not used, and this repository trains from
  scratch.

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

[Terminal Velocity Matching](https://github.com/lumalabs/tvm) for the method and the released code that settles the
details the paper leaves open. [DiT](https://github.com/facebookresearch/DiT) for the backbone it modifies.
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009) for the data. And
[lucid-dl](https://github.com/lucid-official/lucid) for the Apple Silicon runtime.

No model weights or datasets are redistributed here. CIFAR-10 is downloaded at runtime.

## License

Code is MIT.

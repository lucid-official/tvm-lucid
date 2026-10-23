r"""The layers of TVM's semi-Lipschitz DiT, each with its derivative along ``s``.

**Why the derivative is written out.**  TVM's prediction is the terminal
velocity :math:`\frac{d}{ds} f_\theta = F_\theta + (s - t)\,\partial_s F_\theta`,
and the loss back-propagates *through* :math:`\partial_s F_\theta`.  That
needs a Jacobian-vector product whose result stays on the autograd graph.
``lucid.func.jvp`` in lucid-dl 3.15.5 returns its tangent detached — fine
for MeanFlow, whose JVP sits under a stop-gradient, not for TVM — so each
layer here carries its own forward-mode rule: a ``jvp`` method mapping
``(input, input tangent)`` to ``(output, output tangent)``, built from the
same differentiable ops as ``forward``.  ``backward()`` then differentiates
the tangent like any other activation, which is exactly "the gradient goes
through the JVP".  It is also cheaper than a JVP by double backward: one
forward sweep carrying a second stream, and a first-order backward.

**Where the tangent lives.**  Only the end time ``s`` is perturbed (a unit
tangent along ``s``, Section 7.2).  It enters through the interval
embedding of :math:`t - s`, reaches every block through the adaLN
conditioning, and from there the token stream.  The patch embedding and
the positional table do not depend on ``s``, so the token tangent starts at
zero.

Each rule is the textbook derivative; with :math:`\dot{x}` the tangent,

* linear      :math:`\dot y = W \dot x` (the bias is constant)
* SiLU        :math:`\dot y = \dot x\,\sigma(x)\,(1 + x(1 - \sigma(x)))`
* GELU (tanh) :math:`\dot y = \dot x\,[\tfrac12(1 + \tanh u) + \tfrac12 x\,(1 - \tanh^2 u)\,u']`
* RMSNorm     :math:`\dot y = r^{-1}(\dot x - y\,\overline{y\dot x})`, :math:`r = \sqrt{\overline{x^2} + \epsilon}`
* attention   :math:`\dot A = A \odot (\dot S - \textstyle\sum_j A \odot \dot S)`,
  :math:`\dot o = \dot A V + A \dot V`

and ``tests/test_tvm_method.py`` checks the composite against a central
finite difference of ``forward``.
"""

import math

import lucid
import lucid.linalg
import lucid.nn as nn
import lucid.nn.functional as F
from lucid import Tensor

# Every norm in the released network uses this epsilon: token norms, the
# query/key norms and the modulation norm alike.
NORM_EPS = 1e-6

_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_K = 0.044715


# ── initialisation ───────────────────────────────────────────────────────


def spectral_(weight: Tensor) -> None:
    """Gaussian weight scaled to unit spectral norm (LipsFormer's init, Table 6).

    A convolution kernel counts as the matrix of its output channels against
    everything else — the patch embedding is a linear map on flattened patches.
    """
    with lucid.no_grad():
        nn.init.normal_(weight)
        sigma = lucid.linalg.svdvals(weight.reshape(int(weight.shape[0]), -1))[0]
        weight.div_(sigma)


# ── elementwise and normalisation rules ──────────────────────────────────


def linear_jvp(layer: nn.Linear, x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
    return layer(x), F.linear(dx, layer.weight)


def silu_jvp(x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
    sig = F.sigmoid(x)
    return x * sig, dx * sig * (1.0 + x * (1.0 - sig))


def gelu_tanh_jvp(x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
    """``F.gelu(x, approximate="tanh")`` and its tangent."""
    th = lucid.tanh(_GELU_C * (x + _GELU_K * x * x * x))
    du = _GELU_C * (1.0 + 3.0 * _GELU_K * x * x)
    y = 0.5 * x * (1.0 + th)
    return y, dx * (0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du)


def rms_norm(x: Tensor, weight: Tensor | None = None) -> Tensor:
    """RMS-normalise the last axis (no affine unless ``weight``)."""
    return F.rms_norm(x, (int(x.shape[-1]),), weight=weight, eps=NORM_EPS)


def rms_norm_jvp(x: Tensor, dx: Tensor, weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
    inv = lucid.rsqrt((x * x).mean(dim=-1, keepdim=True) + NORM_EPS)
    y = x * inv
    dy = (dx - y * (y * dx).mean(dim=-1, keepdim=True)) * inv
    if weight is None:
        return y, dy
    return y * weight, dy * weight


def modulation(params: Tensor, count: int, width: int, normalize: bool) -> list[Tensor]:
    """Split an adaLN projection into ``count`` vectors of shape ``(B, 1, width)``.

    With ``normalize`` each vector is RMS-normalised on its own (Eq. 16): an
    adaLN layer's Lipschitz constant grows with its modulation's magnitude,
    which nothing else bounds.
    """
    params = params.reshape(-1, count, width)
    if normalize:
        params = rms_norm(params)
    return [params[:, i : i + 1, :] for i in range(count)]


def modulation_jvp(
    params: Tensor, dparams: Tensor, count: int, width: int, normalize: bool
) -> tuple[list[Tensor], list[Tensor]]:
    params = params.reshape(-1, count, width)
    dparams = dparams.reshape(-1, count, width)
    if normalize:
        params, dparams = rms_norm_jvp(params, dparams)
    return (
        [params[:, i : i + 1, :] for i in range(count)],
        [dparams[:, i : i + 1, :] for i in range(count)],
    )


# ── embeddings ───────────────────────────────────────────────────────────


def _frequencies(value: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """ADM's ladder, which divides by ``half`` (DiT's convention)."""
    half = dim // 2
    ladder = lucid.arange(half, dtype=value.dtype, device=value.device.type)
    return lucid.exp(ladder * (-math.log(max_period) / half))


def sinusoid(value: Tensor, dim: int) -> Tensor:
    """``(B,)`` scalars → ``(B, dim)``: cosines then sines."""
    args = value.reshape(-1, 1) * _frequencies(value, dim).reshape(1, -1)
    return lucid.cat([lucid.cos(args), lucid.sin(args)], dim=-1)


def sinusoid_jvp(value: Tensor, dvalue: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    freqs = _frequencies(value, dim).reshape(1, -1)
    args = value.reshape(-1, 1) * freqs
    dargs = dvalue.reshape(-1, 1) * freqs
    cos, sin = lucid.cos(args), lucid.sin(args)
    return lucid.cat([cos, sin], dim=-1), lucid.cat([-sin * dargs, cos * dargs], dim=-1)


def sincos_position_embedding(dim: int, side: int) -> Tensor:
    """Frozen 2-D sine-cosine table ``(1, side², dim)``, column half first.

    The reference builds its halves from ``numpy.meshgrid(w, h)``, whose first
    output is the column coordinate; the order matters to anything trained
    against it.
    """
    quarter = dim // 4
    omega = 1.0 / (10000.0 ** (lucid.arange(quarter, dtype=lucid.float32) / quarter))
    pos = lucid.arange(side, dtype=lucid.float32)
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    axis = lucid.cat([lucid.sin(out), lucid.cos(out)], dim=1)
    rows = axis.reshape(side, 1, -1) + lucid.zeros((side, side, 1))
    cols = axis.reshape(1, side, -1) + lucid.zeros((side, side, 1))
    return lucid.cat([cols, rows], dim=-1).reshape(1, side * side, dim)


class ScalarEmbedder(nn.Module):
    """One scalar condition: sinusoid, then Linear–SiLU–Linear."""

    def __init__(self, width: int, freq_size: int) -> None:
        super().__init__()
        self.freq_size = freq_size
        self.fc1 = nn.Linear(freq_size, width)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(sinusoid(value, self.freq_size))))

    def jvp(self, value: Tensor, dvalue: Tensor) -> tuple[Tensor, Tensor]:
        h, dh = sinusoid_jvp(value, dvalue, self.freq_size)
        h, dh = linear_jvp(self.fc1, h, dh)
        h, dh = silu_jvp(h, dh)
        return linear_jvp(self.fc2, h, dh)


# ── transformer ──────────────────────────────────────────────────────────


class Attention(nn.Module):
    """Multi-head self-attention with RMS-normalised queries and keys."""

    def __init__(self, width: int, num_heads: int, qk_norm: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        # The L2 QK-norm of LipsFormer — it bounds the attention logits —
        # with a learned per-channel scale, as in the released network.
        self.qk_norm = qk_norm
        self.q_norm: nn.Module = (
            nn.RMSNorm(self.head_dim, eps=NORM_EPS) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            nn.RMSNorm(self.head_dim, eps=NORM_EPS) if qk_norm else nn.Identity()
        )
        self.proj = nn.Linear(width, width)

    def _heads(self, qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(B, N, 3D)`` → three ``(B, H, N, D/H)``."""
        batch, tokens = int(qkv.shape[0]), int(qkv.shape[1])
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    @staticmethod
    def _merge(x: Tensor) -> Tensor:
        """``(B, H, N, D/H)`` → ``(B, N, D)``."""
        batch, heads, tokens, head_dim = (int(d) for d in x.shape)
        return x.permute(0, 2, 1, 3).reshape(batch, tokens, heads * head_dim)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._heads(self.qkv(x))
        q, k = self.q_norm(q), self.k_norm(k)
        return self.proj(self._merge(F.scaled_dot_product_attention(q, k, v)))

    def jvp(self, x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
        qkv, dqkv = linear_jvp(self.qkv, x, dx)
        q, k, v = self._heads(qkv)
        dq, dk, dv = self._heads(dqkv)
        if self.qk_norm:
            q, dq = rms_norm_jvp(q, dq, self.q_norm.weight)
            k, dk = rms_norm_jvp(k, dk, self.k_norm.weight)
        scale = 1.0 / math.sqrt(self.head_dim)
        kt, dkt = k.transpose(-2, -1), dk.transpose(-2, -1)
        probs = F.softmax((q @ kt) * scale, dim=-1)
        dlogits = (dq @ kt + q @ dkt) * scale
        dprobs = probs * (dlogits - (probs * dlogits).sum(dim=-1, keepdim=True))
        out = probs @ v
        dout = dprobs @ v + probs @ dv
        return linear_jvp(self.proj, self._merge(out), self._merge(dout))


class FeedForward(nn.Module):
    """DiT's MLP: Linear, tanh-approximated GELU, Linear."""

    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, inner)
        self.fc2 = nn.Linear(inner, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    def jvp(self, x: Tensor, dx: Tensor) -> tuple[Tensor, Tensor]:
        h, dh = linear_jvp(self.fc1, x, dx)
        h, dh = gelu_tanh_jvp(h, dh)
        return linear_jvp(self.fc2, h, dh)


class Block(nn.Module):
    r"""A DiT block with every LayerNorm and every modulation RMS-normalised.

    Appendix B.2: :math:`x \leftarrow x + c_1 \cdot \mathrm{ATTN}(\mathrm{RMSNorm}(x)
    (1 + a_1) + b_1)`, then the same with the MLP — DiT's adaLN-Zero with
    parameter-free norms.  The released network keeps DiT's ``1 +`` on the
    scale, and so does this block.
    """

    def __init__(
        self, width: int, num_heads: int, mlp_ratio: float, qk_norm: bool, modulation_norm: bool
    ) -> None:
        super().__init__()
        self.modulation_norm = modulation_norm
        self.attn = Attention(width, num_heads, qk_norm)
        self.mlp = FeedForward(width, int(width * mlp_ratio))
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        width = int(x.shape[-1])
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation(
            self.ada_ln(cond), 6, width, self.modulation_norm
        )
        x = x + gate_a * self.attn(rms_norm(x) * (1.0 + scale_a) + shift_a)
        return x + gate_m * self.mlp(rms_norm(x) * (1.0 + scale_m) + shift_m)

    def jvp(self, x: Tensor, dx: Tensor, act: Tensor, dact: Tensor) -> tuple[Tensor, Tensor]:
        """``act``/``dact`` are ``SiLU(cond)`` and its tangent, shared by every block."""
        width = int(x.shape[-1])
        mod, dmod = linear_jvp(self.ada_ln[1], act, dact)
        (sa, ca, ga, sm, cm, gm), (dsa, dca, dga, dsm, dcm, dgm) = modulation_jvp(
            mod, dmod, 6, width, self.modulation_norm
        )
        h, dh = rms_norm_jvp(x, dx)
        a, da = self.attn.jvp(h * (1.0 + ca) + sa, dh * (1.0 + ca) + h * dca + dsa)
        x, dx = x + ga * a, dx + dga * a + ga * da
        h, dh = rms_norm_jvp(x, dx)
        m, dm = self.mlp.jvp(h * (1.0 + cm) + sm, dh * (1.0 + cm) + h * dcm + dsm)
        return x + gm * m, dx + dgm * m + gm * dm


class FinalLayer(nn.Module):
    """Modulated RMSNorm, then the projection back to patch pixels."""

    def __init__(
        self, width: int, patch_size: int, out_channels: int, modulation_norm: bool
    ) -> None:
        super().__init__()
        self.modulation_norm = modulation_norm
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.proj = nn.Linear(width, patch_size * patch_size * out_channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = modulation(self.ada_ln(cond), 2, int(x.shape[-1]), self.modulation_norm)
        return self.proj(rms_norm(x) * (1.0 + scale) + shift)

    def jvp(self, x: Tensor, dx: Tensor, act: Tensor, dact: Tensor) -> tuple[Tensor, Tensor]:
        mod, dmod = linear_jvp(self.ada_ln[1], act, dact)
        (shift, scale), (dshift, dscale) = modulation_jvp(
            mod, dmod, 2, int(x.shape[-1]), self.modulation_norm
        )
        h, dh = rms_norm_jvp(x, dx)
        return linear_jvp(
            self.proj, h * (1.0 + scale) + shift, dh * (1.0 + scale) + h * dscale + dshift
        )

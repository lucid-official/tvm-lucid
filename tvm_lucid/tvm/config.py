"""Every choice the paper makes, with its default.

Defaults are the ImageNet-256 recipe — Table 6, column 1, and the released
``im256_w2`` configuration — so a field left alone means "as the paper".
``tvm_lucid.model`` overrides the data-shaped ones (size, channels,
classes) for CIFAR-10 and the TOML ``[model]`` section can override any
other.
"""

from dataclasses import dataclass
from typing import Literal

# Joint distribution of (t, s) — Section 4 and Table 4a.
#   gap   : gap ~ logit-normal, s | gap ~ logit-normal truncated to [0, 1 − gap],
#           t = s + gap.  The paper's gap*, its default.
#   trunc : t ~ logit-normal, s | t ~ logit-normal truncated to [0, t].
#   clamp : t, s independent logit-normals, s = min(s, t).
TimeSampler = Literal["gap", "trunc", "clamp"]

# How the scalar-embedding MLPs start (Table 6, "Time Embed init").
TimeEmbeddingInit = Literal["spectral", "normal"]


@dataclass(frozen=True)
class TVMConfig:
    """Network, objective and sampler of Terminal Velocity Matching.

    Network (Section 4, Appendix B.2)
        sample_size, in_channels, out_channels, patch_size, hidden_size,
        depth, num_heads, mlp_ratio, frequency_embedding_size
            DiT's shape.  XL/2 is 1152 wide, 28 deep, with 18 heads in the
            released checkpoints (DiT ships 16).
        qk_norm
            RMS-normalise each head's queries and keys (learned scale).
        modulation_norm
            RMS-normalise every adaLN shift, scale and gate (Eq. 16).
        time_embedding_init
            ``"spectral"`` or DiT's ``"normal"`` N(0, 0.02).

    Conditioning
        num_classes
            The label table has one more row: the null label.
        class_dropout
            Probability a label is replaced by the null one; that sample
            trains at w = 1.
        guidance_scale
            The CFG weight w trained at, and sampled at by default.
        guidance_range
            ``(w_min, w_max)`` to train at random weights instead:
            1/w ~ U(1/w_max, 1/w_min).
        scaled_parameterization
            The network predicts F with w factored out; its velocity is
            w·F (Table 4c).

    Times (Section 4, Table 4a)
        time_sampler, gap_mean/std, t_mean/std, s_mean/std
            Logit-normal parameters; see ``TimeSampler`` above.
        separate_flow_matching_time
            Draw the Flow Matching term's time independently (gap*).
        ratio_t_equals_s
            Fraction of samples forced to s = t (pure Flow Matching).

    Training (Algorithm 1)
        target_ema_decay
            γ of θ*, which the targets are read from (Table 4b: 0.99).
            0 uses the live weights under a stop-gradient.
        eval_ema_decay
            γ of θ**, the network evaluated and released.  ``None`` keeps
            none.
        detach_jvp
            Stop the gradient at ∂F/∂s (an ablation; the paper does not).
    """

    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 18
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256
    qk_norm: bool = True
    modulation_norm: bool = True
    time_embedding_init: TimeEmbeddingInit = "spectral"

    num_classes: int = 1000
    class_dropout: float = 0.1
    guidance_scale: float = 2.0
    guidance_range: tuple[float, float] | None = None
    scaled_parameterization: bool = True

    time_sampler: TimeSampler = "gap"
    gap_mean: float = -0.8
    gap_std: float = 1.0
    t_mean: float = 1.0
    t_std: float = 1.0
    s_mean: float = -0.4
    s_std: float = 1.0
    separate_flow_matching_time: bool = True
    ratio_t_equals_s: float = 0.0

    target_ema_decay: float = 0.99
    eval_ema_decay: float | None = 0.9999
    detach_jvp: bool = False

    def __post_init__(self) -> None:
        if self.sample_size % self.patch_size:
            raise ValueError(
                f"sample_size {self.sample_size} is not a multiple of patch_size {self.patch_size}"
            )
        if self.hidden_size % 4:
            raise ValueError(
                f"hidden_size must be divisible by 4 for the 2-D sine-cosine table, "
                f"got {self.hidden_size}"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        if self.frequency_embedding_size % 2:
            raise ValueError(
                f"frequency_embedding_size must be even (cosine and sine halves), "
                f"got {self.frequency_embedding_size}"
            )
        if self.time_sampler not in ("gap", "trunc", "clamp"):
            raise ValueError(f"unknown time_sampler {self.time_sampler!r}")
        if self.time_embedding_init not in ("spectral", "normal"):
            raise ValueError(f"unknown time_embedding_init {self.time_embedding_init!r}")
        if self.guidance_scale <= 0.0:
            raise ValueError(f"guidance_scale must be positive, got {self.guidance_scale}")
        if self.guidance_range is not None:
            low, high = self.guidance_range
            if not 0.0 < low <= high:
                raise ValueError(
                    f"guidance_range must satisfy 0 < low <= high, got {self.guidance_range}"
                )
        for name in ("class_dropout", "ratio_t_equals_s"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} is a probability, got {getattr(self, name)}")
        for name in ("gap_std", "t_std", "s_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.target_ema_decay < 1.0:
            raise ValueError(f"target_ema_decay must be in [0, 1), got {self.target_ema_decay}")
        if self.eval_ema_decay is not None and not 0.0 < self.eval_ema_decay < 1.0:
            raise ValueError(f"eval_ema_decay must be in (0, 1), got {self.eval_ema_decay}")

    @property
    def grid(self) -> int:
        """Tokens along one side of the image."""
        return self.sample_size // self.patch_size

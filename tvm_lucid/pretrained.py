"""The released ImageNet checkpoints (``lumaai/tvm``) in :class:`TVMNet`.

``scripts/convert_released.py`` turns a release pickle into SafeTensors under
the upstream parameter names; :func:`load` reads that file with
``lucid.serialization.load_safetensors`` and renames each tensor onto this
port.  The mapping is total in both directions — every upstream tensor has
one home here and every parameter here is filled — and the frozen
positional table, which the release does store, is checked against the one
this port rebuilds rather than copied.

All three releases are DiT-XL/2 with 18 heads (678M parameters) over SD-VAE
latents (``stabilityai/sd-vae-ft-ema``), whitened per channel with the
statistics below.  They differ in resolution, guidance weight, and whether
the network was trained with ``w`` factored out (``do_scale_param``).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import lucid
import lucid.serialization
from lucid import Tensor

from tvm_lucid.tvm import TVMConfig, TVMNet


@dataclass(frozen=True)
class Release:
    """One released checkpoint and what its config file says about it."""

    name: str
    image_size: int
    guidance_scale: float
    scaled_parameterization: bool  # ``network.do_scale_param``
    time_embedding_init: str  # ``time_specinit``; recorded, irrelevant once loaded
    raw_mean: tuple[float, float, float, float]  # ``encoder.raw_mean``
    raw_std: tuple[float, float, float, float]  # ``encoder.raw_std``; final_std is 1
    fid: dict[int, float]  # NFE → FID-50k, as reported with the release

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

    def config(self) -> TVMConfig:
        return TVMConfig(
            sample_size=self.latent_size,
            in_channels=4,
            out_channels=4,
            num_classes=1000,
            guidance_scale=self.guidance_scale,
            scaled_parameterization=self.scaled_parameterization,
            time_embedding_init=self.time_embedding_init,  # type: ignore[arg-type]
        )


_IM256_STATS = dict(
    raw_mean=(0.86488, -0.27787343, 0.21616915, 0.3738409),
    raw_std=(4.85503674, 5.31922414, 3.93725398, 3.9870003),
)
_IM512_STATS = dict(raw_mean=(1.56, -0.695, 0.483, 0.729), raw_std=(5.27, 5.91, 4.21, 4.31))

RELEASES = {
    r.name: r
    for r in (
        Release("im256_w2", 256, 2.0, True, "spectral", fid={1: 3.29}, **_IM256_STATS),
        Release("im256_w1p75", 256, 1.75, True, "normal", fid={4: 1.99}, **_IM256_STATS),
        Release("im512_w2p5", 512, 2.5, False, "normal", fid={1: 4.32}, **_IM512_STATS),
    )
}

# upstream name (``model.`` stripped) → TVMNet name, by pattern.
_RENAMES = [
    (r"x_embedder\.proj\.", "patch_embed."),
    (r"t_embedder\.mlp\.0\.", "time_embed.fc1."),
    (r"t_embedder\.mlp\.2\.", "time_embed.fc2."),
    (r"s_embedder\.mlp\.0\.", "interval_embed.fc1."),
    (r"s_embedder\.mlp\.2\.", "interval_embed.fc2."),
    (r"guidance_embedder\.mlp\.0\.", "guidance_embed.fc1."),
    (r"guidance_embedder\.mlp\.2\.", "guidance_embed.fc2."),
    (r"y_embedder\.embedding_table\.", "label_embed."),
    (r"(blocks\.\d+)\.adaLN_modulation\.1\.", r"\1.ada_ln.1."),
    (r"(blocks\.\d+\.(attn|mlp))\.", r"\1."),  # qkv, q_norm, k_norm, proj; fc1, fc2
    (r"final_layer\.linear\.", "final.proj."),
    (r"final_layer\.adaLN_modulation\.1\.", "final.ada_ln.1."),
]


def rename(upstream: str) -> str:
    name = upstream.removeprefix("model.")
    for pattern, replacement in _RENAMES:
        renamed, hits = re.subn(f"^{pattern}", replacement, name)
        if hits:
            return renamed
    raise KeyError(f"no TVMNet parameter for upstream {upstream!r}")


def load(path: str, config: TVMConfig, *, device: str = "metal") -> TVMNet:
    """The released network, in eval mode, from a converted SafeTensors file.

    ``config`` is :meth:`Release.config` for a release.
    """
    state = lucid.serialization.load_safetensors(path, device=device)
    pos_embed = state.pop("model.pos_embed")
    renamed: dict[str, Tensor] = {rename(k): v for k, v in state.items()}  # type: ignore[misc]
    net = TVMNet(config).to(device).eval()
    expected = set(net.state_dict())
    if set(renamed) != expected:
        missing, extra = sorted(expected - set(renamed)), sorted(set(renamed) - expected)
        raise KeyError(f"state mismatch: missing {missing[:5]}, unexpected {extra[:5]}")
    net.load_state_dict(renamed, strict=True)
    # The release stores its frozen sine-cosine table; this port rebuilds it.
    gap = float((pos_embed.to(device) - net.pos_embed).abs().max().item())
    if gap > 1e-5:
        raise ValueError(f"positional table differs from the release's by {gap:.2e}")
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def sample(
    net,  # type: ignore[no-untyped-def]  # TVMNet, or anything called like one
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int,
    guidance_scale: float,
    on_step: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    r"""Figure 3's sampler: ``steps`` even jumps :math:`x_s = x_t + (s - t)\,u(x_t, t, s)`.

    Every input is a tensor, so a compiled or exported ``net`` sees one
    signature per batch size.  Upstream integrates in float64; Metal has
    no float64, and the sum here is ``steps`` terms long.  ``on_step(k, x)``
    is called after jump ``k`` of ``steps``; on Metal ``x`` may not have been
    computed yet, so a caller timing it must read it first.
    """
    n, device, dtype = int(noise.shape[0]), noise.device.type, noise.dtype
    guidance = lucid.full((n,), guidance_scale, device=device, dtype=dtype)
    x = noise
    with lucid.no_grad():
        for i in range(steps):
            t_value, s_value = 1.0 - i / steps, 1.0 - (i + 1) / steps
            t = lucid.full((n,), t_value, device=device, dtype=dtype)
            s = lucid.full((n,), s_value, device=device, dtype=dtype)
            x = x + (s_value - t_value) * net(x, t, s, labels, guidance)
            if on_step is not None:
                on_step(i + 1, x)
    return x


def to_vae_latents(x: Tensor, release: Release) -> Tensor:
    """Whitened model latents → the raw SD-VAE latents the decoder takes."""
    std = lucid.tensor(list(release.raw_std), device=x.device.type, dtype=x.dtype)
    mean = lucid.tensor(list(release.raw_mean), device=x.device.type, dtype=x.dtype)
    return x * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)

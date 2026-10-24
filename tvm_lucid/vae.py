"""The decoder half of Stable Diffusion's KL autoencoder, for the released latents.

The ImageNet checkpoints generate 4-channel latents of
``stabilityai/sd-vae-ft-ema`` (8x downsampling).  Only decoding is needed,
so only the decoder is built: ``post_quant_conv``, then conv-in, a middle
block (ResNet, single-head self-attention, ResNet) and four up blocks of
three ResNets each, the first three ending in a 2x nearest upsample.
Every GroupNorm has 32 groups and ε = 1e-6.

Module names follow the diffusers checkpoint, so its SafeTensors file loads
by name; :func:`load` also accepts the newer attention names (``to_q`` …).
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.serialization
from lucid import Tensor

_GROUPS, _EPS = 32, 1e-6
_CHANNELS = (128, 256, 512, 512)  # ``block_out_channels``; the decoder runs them backwards
_LAYERS = 2  # ``layers_per_block``; a decoder block has one more


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_GROUPS, in_ch, eps=_EPS)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_GROUPS, out_ch, eps=_EPS)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class SelfAttention(nn.Module):
    """One head over every pixel: GroupNorm, then attention across ``H·W`` positions."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(_GROUPS, ch, eps=_EPS)
        self.query = nn.Linear(ch, ch)
        self.key = nn.Linear(ch, ch)
        self.value = nn.Linear(ch, ch)
        self.proj_attn = nn.Linear(ch, ch)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = (int(d) for d in x.shape)
        seq = self.group_norm(x).reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        q, k, v = (proj(seq).reshape(b, 1, h * w, c) for proj in (self.query, self.key, self.value))
        out = F.scaled_dot_product_attention(q, k, v).reshape(b, h * w, c)
        return x + self.proj_attn(out).permute(0, 2, 1).reshape(b, c, h, w)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class MidBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(ch, ch), ResnetBlock(ch, ch)])
        self.attentions = nn.ModuleList([SelfAttention(ch)])

    def forward(self, x: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(_LAYERS + 1)]
        )
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if upsample else None

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return x if self.upsamplers is None else self.upsamplers[0](x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        widths = list(reversed(_CHANNELS))
        self.conv_in = nn.Conv2d(latent_channels, widths[0], 3, padding=1)
        self.mid_block = MidBlock(widths[0])
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(widths[max(i - 1, 0)], widths[i], upsample=i < len(widths) - 1)
                for i in range(len(widths))
            ]
        )
        self.conv_norm_out = nn.GroupNorm(_GROUPS, widths[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(widths[-1], 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class VAEDecoder(nn.Module):
    """Raw SD-VAE latents ``(B, 4, h, w)`` → images ``(B, 3, 8h, 8w)`` in about [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.decoder = Decoder()

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(z))


_NEW_ATTENTION_NAMES = {
    ".to_q.": ".query.",
    ".to_k.": ".key.",
    ".to_v.": ".value.",
    ".to_out.0.": ".proj_attn.",
}


def load(path: str, *, device: str = "metal") -> VAEDecoder:
    """The decoder from a diffusers ``AutoencoderKL`` SafeTensors file (encoder ignored)."""
    state = lucid.serialization.load_safetensors(path, device=device)
    wanted: dict[str, Tensor] = {}
    for name, value in state.items():
        if not name.startswith(("decoder.", "post_quant_conv.")):
            continue
        for new, old in _NEW_ATTENTION_NAMES.items():
            name = name.replace(new, old)
        if value.ndim == 4 and ".attentions." in name:  # 1x1 convs in some exports
            value = value.reshape(int(value.shape[0]), int(value.shape[1]))
        wanted[name] = value  # type: ignore[assignment]
    vae = VAEDecoder().to(device).eval()
    vae.load_state_dict(wanted, strict=True)
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def to_pixels(images: Tensor) -> Tensor:
    """Decoder output → ``(B, H, W, 3)`` integer values in [0, 255], truncated as upstream does."""
    return ((images * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).floor().permute(0, 2, 3, 1)

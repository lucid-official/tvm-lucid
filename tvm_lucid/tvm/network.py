r"""The two-time velocity network :math:`w\,F_\theta(x_t, t, s, c, w)`.

DiT with its Lipschitz holes closed (Section 4, Appendix B.2): patchify,
add a frozen 2-D sine-cosine table, run adaLN-Zero blocks over a summed
conditioning vector, unpatchify.  Four things condition the field — the
start time :math:`t`, the interval :math:`t - s`, the reciprocal guidance
weight :math:`1/w`, and the class — each a sinusoid through its own MLP
(the label a table lookup), summed.

The output is the **average velocity over** :math:`[s, t]`, so one sampler
step is :math:`x_s = x_t + (s - t)\,\mathrm{out}`; at :math:`s = t` it is the
instantaneous velocity.  Under the scaled parameterisation it is
:math:`w F_\theta`.

Parameter names match ``lucid.models.generative.tvm.TVMModel`` from Lucid's
development branch, so a state dict moves between the two unchanged.
"""

import lucid
import lucid.autograd
import lucid.nn as nn
from lucid import Tensor

from tvm_lucid.tvm.config import TVMConfig
from tvm_lucid.tvm.layers import (
    Block,
    FinalLayer,
    ScalarEmbedder,
    silu,
    silu_jvp,
    sincos_position_embedding,
    spectral_,
)


class TVMNet(nn.Module):
    def __init__(self, config: TVMConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.patch_embed = nn.Conv2d(
            config.in_channels, width, kernel_size=config.patch_size, stride=config.patch_size
        )
        # Rebuilt from the config, never saved — frozen in the reference too.
        self.register_buffer(
            "pos_embed", sincos_position_embedding(width, config.grid), persistent=False
        )
        freq = config.frequency_embedding_size
        self.time_embed = ScalarEmbedder(width, freq)
        self.interval_embed = ScalarEmbedder(width, freq)
        self.guidance_embed = ScalarEmbedder(width, freq)
        self.label_embed = nn.Embedding(config.num_classes + 1, width)  # last row: null label
        self.blocks = nn.ModuleList(
            [
                Block(
                    width,
                    config.num_heads,
                    config.mlp_ratio,
                    config.qk_norm,
                    config.modulation_norm,
                )
                for _ in range(config.depth)
            ]
        )
        self.final = FinalLayer(
            width, config.patch_size, config.out_channels, config.modulation_norm
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Lipschitz initialisation, then adaLN-Zero (Table 6).

        Linear layers that are not zeroed start at unit spectral norm; the
        scalar embedders follow ``time_embedding_init``; the label table is
        DiT's :math:`\mathcal{N}(0, 0.02)`.  Modulations and the decoder start
        at zero, so every block starts as the identity and the field at
        zero.  Biases start at zero.
        """
        spectral_time = self.config.time_embedding_init == "spectral"
        with lucid.no_grad():
            spectral_(self.patch_embed.weight)
            for block in self.blocks:
                for linear in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2):
                    spectral_(linear.weight)
                block.ada_ln[1].weight.zero_()
            for embedder in (self.time_embed, self.interval_embed, self.guidance_embed):
                for linear in (embedder.fc1, embedder.fc2):
                    if spectral_time:
                        spectral_(linear.weight)
                    else:
                        nn.init.normal_(linear.weight, std=0.02)
            nn.init.normal_(self.label_embed.weight, std=0.02)
            self.final.ada_ln[1].weight.zero_()
            self.final.proj.weight.zero_()
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                    module.bias.zero_()

    # ── inputs ───────────────────────────────────────────────────────────

    def _resolve(
        self, x: Tensor, labels: Tensor | None, guidance: Tensor | float | None
    ) -> tuple[Tensor, Tensor]:
        """Per-sample labels and guidance weights.

        No labels → the null label.  No guidance → the trained weight for a
        labelled call, 1 for an unlabelled one (the only weight the null
        label ever trains at).
        """
        batch, device = int(x.shape[0]), x.device.type
        if guidance is None:
            guidance = 1.0 if labels is None else self.config.guidance_scale
        if not isinstance(guidance, Tensor):
            guidance = lucid.full((batch,), float(guidance), device=device, dtype=x.dtype)
        if labels is None:
            labels = lucid.full((batch,), self.config.num_classes, device=device, dtype=lucid.int64)
        return labels, guidance

    def _tokens(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x)
        tokens = tokens.reshape(int(tokens.shape[0]), int(tokens.shape[1]), -1).permute(0, 2, 1)
        return tokens + self.pos_embed

    def _unpatchify(self, x: Tensor) -> Tensor:
        """``(B, N, p²C)`` → ``(B, C, H, W)``."""
        p, c, g = self.config.patch_size, self.config.out_channels, self.config.grid
        x = x.reshape(-1, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(-1, c, g * p, g * p)

    def _checkpointed(self, segment, *inputs: Tensor):  # type: ignore[no-untyped-def]
        """``segment(*inputs)``, recomputed in backward under ``gradient_checkpointing``.

        Only while autograd records: under ``no_grad`` (target reads, sampling)
        nothing is kept anyway.  Reentrant mode is enough — the token stream
        always requires grad here, so the segment is always recomputed.
        """
        if self.config.gradient_checkpointing and lucid.is_grad_enabled():
            return lucid.autograd.checkpoint(
                segment, *inputs, use_reentrant=True, preserve_rng_state=False
            )
        return segment(*inputs)

    def _scale(self, out: Tensor, guidance: Tensor) -> Tensor:
        if self.config.scaled_parameterization:
            return out * guidance.reshape(-1, 1, 1, 1)
        return out

    # ── evaluation ───────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        s: Tensor,
        labels: Tensor | None = None,
        guidance: Tensor | float | None = None,
    ) -> Tensor:
        """Average velocity from ``t`` to ``s`` (``(B,)`` each, ``s <= t``) at ``x``."""
        labels, guidance = self._resolve(x, labels, guidance)
        cond = (
            self.time_embed(t)
            + self.interval_embed(t - s)
            + self.guidance_embed(1.0 / guidance)
            + self.label_embed(labels)
        )
        act = silu(cond)
        tokens = self._tokens(x)
        for block in self.blocks:
            tokens = self._checkpointed(block, tokens, act)
        return self._scale(self._unpatchify(self.final(tokens, act)), guidance)

    def jvp_s(
        self,
        x: Tensor,
        t: Tensor,
        s: Tensor,
        labels: Tensor | None = None,
        guidance: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""The forward output and its derivative along ``s``: :math:`(u, \partial_s u)`.

        One sweep carrying a second stream, the tangent of every activation
        with respect to ``s`` (unit tangent per sample).  Both outputs stay on
        the autograd graph.
        """
        labels, guidance = self._resolve(x, labels, guidance)
        interval, dinterval = self.interval_embed.jvp(t - s, -lucid.ones_like(s))
        cond = (
            self.time_embed(t)
            + interval
            + self.guidance_embed(1.0 / guidance)
            + self.label_embed(labels)
        )
        act, dact = silu_jvp(cond, dinterval)
        tokens = self._tokens(x)
        dtokens = lucid.zeros_like(tokens)
        for block in self.blocks:
            tokens, dtokens = self._checkpointed(block.jvp, tokens, dtokens, act, dact)
        out, dout = self.final.jvp(tokens, dtokens, act, dact)
        return (
            self._scale(self._unpatchify(out), guidance),
            self._scale(self._unpatchify(dout), guidance),
        )

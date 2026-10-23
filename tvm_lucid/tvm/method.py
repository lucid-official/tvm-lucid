r"""Terminal Velocity Matching: the objective (Eq. 15), Algorithm 1's two EMAs,
and the few-step sampler (Figure 3).

Per sample, with :math:`\beta = 1/w` and :math:`x_t = (1 - t)\,x + t\,\epsilon`,

.. math::

    \beta^2\Bigl\lVert \underbrace{u + (s - t)\,\partial_s u}_{\frac{d}{ds} f_\theta}
      - u_{\theta^*}\bigl(\mathrm{sg}[x_t + (s - t)\,u],\ s\bigr)\Bigr\rVert^2
      \mathbb{1}_{t \ne s}
    \;+\; \beta^2\bigl\lVert u_\theta(x_{s'}, s')
      - [\,w\,(\epsilon - x) + (1 - w)\,u_{\theta^*}(x_{s'}, s', \varnothing, 1)\,]\bigr\rVert^2

where :math:`u = w F_\theta(x_t, t, s)` is the network's average velocity.
The first term trains the jump's *terminal velocity* — its derivative at
the landing time — to match the field at the landing point; the second
is classifier-free-guided Flow Matching at :math:`s = t`, which pins that
field down.  Targets come from :math:`\theta^*`, an EMA of the weights,
under a stop-gradient; the gradient flows through :math:`\partial_s u`.

Squared norms are averaged over elements, as the released code does (the
paper's sum differs by the constant :math:`CHW`).
"""

from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.special
from lucid import Tensor

from tvm_lucid.tvm.config import TVMConfig
from tvm_lucid.tvm.network import TVMNet


@dataclass
class Losses:
    loss: Tensor
    """Eq. 15, a scalar: the two terms summed per sample, averaged over the batch."""
    terminal: Tensor
    """Batch mean of the terminal-velocity term (detached); zero where s = t."""
    flow_matching: Tensor
    """Batch mean of the guided Flow Matching term (detached)."""


class TVM(nn.Module):
    r"""The network being trained (``net``) and everything Algorithm 1 does with it.

    **Call** :meth:`update_ema` **after every optimiser step.**  Two EMAs of
    ``net`` are kept: :math:`\theta^*` (``target_network``, rate
    ``target_ema_decay``) supplies the targets; :math:`\theta^{**}`
    (``ema_network``, rate ``eval_ema_decay``) is what gets evaluated.  Both
    start as copies of the live weights the first time either is asked for
    — the first training step does — and live outside the module tree:
    not parameters, not in ``state_dict()``, invisible to the optimiser.
    Checkpoint them through the two properties.
    """

    def __init__(self, config: TVMConfig) -> None:
        super().__init__()
        self.config = config
        self.net = TVMNet(config)
        # A plain dict: a module attribute would register the copies as
        # children — parameters for the optimiser, entries in state_dict.
        self._averages: dict[str, TVMNet] = {}

    # ── Algorithm 1's two averages ───────────────────────────────────────

    def _decays(self) -> dict[str, float]:
        decays: dict[str, float] = {}
        if self.config.target_ema_decay > 0.0:
            decays["target"] = self.config.target_ema_decay
        if self.config.eval_ema_decay is not None:
            decays["eval"] = self.config.eval_ema_decay
        return decays

    def _average(self, name: str) -> TVMNet:
        device = next(iter(self.net.parameters())).device.type
        if not self._averages:
            live = dict(self.net.named_parameters())
            for key in self._decays():
                # Onto the live weights' device first: copy_ does not cross devices.
                copy = TVMNet(self.config).to(device).eval()
                with lucid.no_grad():
                    for param_name, param in copy.named_parameters():
                        param.requires_grad_(False)
                        param.copy_(live[param_name].detach())
                self._averages[key] = copy
        average = self._averages[name]
        if next(iter(average.parameters())).device.type != device:
            average.to(device)
        return average

    @property
    def target_network(self) -> TVMNet:
        """θ*, or the live network itself when ``target_ema_decay`` is 0 (Table 4b, γ = 0)."""
        if "target" not in self._decays():
            return self.net
        return self._average("target")

    @property
    def ema_network(self) -> TVMNet:
        """θ**, the network Algorithm 1 outputs."""
        if "eval" not in self._decays():
            raise ValueError("eval_ema_decay is None: no evaluation EMA is kept")
        return self._average("eval")

    def update_ema(self) -> None:
        r""":math:`\theta^{(*)} \leftarrow \gamma\,\theta^{(*)} + (1 - \gamma)\,\theta`, both averages.

        Paired by name: two identically built networks need not list their
        parameters in the same order once anything has been re-assigned.

        Ends by evaluating the averages (and with them the live weights they
        read).  On Metal every op is lazy, and θ** is not read again until the
        next evaluation, so without the flush each step would chain one more
        update onto it — holding every past copy of the weights alive, and
        making each step slower than the last.
        """
        live = dict(self.net.named_parameters())
        updated: list[Tensor] = []
        with lucid.no_grad():
            for key, decay in self._decays().items():
                for name, slow in self._average(key).named_parameters():
                    slow.copy_(slow * decay + live[name].detach() * (1.0 - decay))
                    updated.append(slow)
        lucid.eval(*updated, *live.values())

    # ── training variables ───────────────────────────────────────────────

    @staticmethod
    def _logit_normal(mean: float, std: float, batch: int, device: str) -> Tensor:
        return F.sigmoid(lucid.randn((batch,), device=device) * std + mean)

    @staticmethod
    def _truncated_logit_normal(mean: float, std: float, upper: Tensor, device: str) -> Tensor:
        r"""Logit-normal draws conditioned on lying below ``upper``, by inverse CDF.

        A uniform scaled into :math:`[0, \Phi((\mathrm{logit}(\mathrm{upper}) - \mu)/\sigma)]`
        and mapped back through :math:`\Phi^{-1}` and the logistic: exact, no
        rejection, nothing above the bound.
        """
        logit_upper = lucid.log(upper / (1.0 - upper))
        mass = lucid.special.ndtr((logit_upper - mean) / std)
        uniform = lucid.rand((int(upper.shape[0]),), device=device) * mass
        return F.sigmoid(lucid.special.ndtri(uniform) * std + mean)

    def sample_times(self, batch: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
        """``t``, ``s <= t`` for the terminal term, and the Flow Matching time."""
        c = self.config
        if c.time_sampler == "gap":
            gap = self._logit_normal(c.gap_mean, c.gap_std, batch, device)
            s = self._truncated_logit_normal(c.s_mean, c.s_std, 1.0 - gap, device)
            t = lucid.clamp(s + gap, max=1.0)
        elif c.time_sampler == "trunc":
            t = self._logit_normal(c.t_mean, c.t_std, batch, device)
            s = self._truncated_logit_normal(c.s_mean, c.s_std, t, device)
        else:
            t = self._logit_normal(c.t_mean, c.t_std, batch, device)
            s = lucid.minimum(self._logit_normal(c.s_mean, c.s_std, batch, device), t)
        if c.ratio_t_equals_s > 0.0:
            s = lucid.where(lucid.rand((batch,), device=device) < c.ratio_t_equals_s, t, s)
        if c.separate_flow_matching_time:
            s_fm = self._logit_normal(c.s_mean, c.s_std, batch, device)
        else:
            s_fm = s
        return t, s, s_fm

    def sample_guidance(self, batch: int, device: str) -> Tensor:
        """Constant w, or 1/w ~ U(1/w_max, 1/w_min) under ``guidance_range``."""
        c = self.config
        if c.guidance_range is None:
            return lucid.full((batch,), c.guidance_scale, device=device)
        low, high = c.guidance_range
        beta = lucid.rand((batch,), device=device) * (1.0 / low - 1.0 / high) + 1.0 / high
        return 1.0 / beta

    # ── objective ────────────────────────────────────────────────────────

    def _terminal_velocity(
        self, x_t: Tensor, t: Tensor, s: Tensor, labels: Tensor, guidance: Tensor
    ) -> tuple[Tensor, Tensor]:
        r""":math:`u = wF_\theta(x_t, t, s)` and :math:`\frac{d}{ds} f_\theta = u + (s - t)\,\partial_s u`."""
        if self.config.detach_jvp:
            velocity = self.net(x_t, t, s, labels, guidance)
            with lucid.no_grad():
                _, along_s = self.net.jvp_s(x_t, t, s, labels, guidance)
        else:
            velocity, along_s = self.net.jvp_s(x_t, t, s, labels, guidance)
        return velocity, velocity + (s - t).reshape(-1, 1, 1, 1) * along_s

    def forward(self, images: Tensor, labels: Tensor | None = None) -> Losses:
        """One evaluation of Eq. 15 on a batch of data in ``[-1, 1]``."""
        c = self.config
        batch, device = int(images.shape[0]), images.device.type

        t, s, s_fm = self.sample_times(batch, device)
        guidance = self.sample_guidance(batch, device)
        null = lucid.full((batch,), c.num_classes, device=device, dtype=lucid.int64)
        if labels is None:
            labels, guidance = null, lucid.ones_like(guidance)
        else:
            null = null.to(labels.dtype)
            if c.class_dropout > 0.0:
                drop = lucid.rand((batch,), device=device) < c.class_dropout
                labels = lucid.where(drop, null, labels)
                guidance = lucid.where(drop, lucid.ones_like(guidance), guidance)

        noise = lucid.randn(tuple(int(d) for d in images.shape), device=device)
        t_b, s_b = t.reshape(-1, 1, 1, 1), s_fm.reshape(-1, 1, 1, 1)
        x_t = (1.0 - t_b) * images + t_b * noise
        x_fm = (1.0 - s_b) * images + s_b * noise
        w_b = guidance.reshape(-1, 1, 1, 1)

        # On Metal every op is lazy, and the objective is evaluated stage by
        # stage: one graph spanning all of it held every stage's activations at
        # once and ran 4x slower (195 s against 44 s per step, S/2, batch 64).
        velocity, terminal = self._terminal_velocity(x_t, t, s, labels, guidance)
        lucid.eval(velocity, terminal)
        landing = (x_t + (s - t).reshape(-1, 1, 1, 1) * velocity).detach()

        # Both targets from θ* in one batch: the field where the jump lands
        # (conditional, at the sample's w) and the unconditional field the
        # guided Flow Matching target mixes in (null label, w = 1).
        with lucid.no_grad():
            reads = self.target_network(
                lucid.cat([landing, x_fm], dim=0),
                lucid.cat([s, s_fm], dim=0),
                lucid.cat([s, s_fm], dim=0),
                lucid.cat([labels, null], dim=0),
                lucid.cat([guidance, lucid.ones_like(guidance)], dim=0),
            )
        lucid.eval(reads)
        terminal_target, unconditional = reads[:batch], reads[batch:]

        velocity_fm = self.net(x_fm, s_fm, s_fm, labels, guidance)
        lucid.eval(velocity_fm)
        guided = w_b * (noise - images) + (1.0 - w_b) * unconditional

        # The released loss drops the terminal term where isclose(s, t) — with
        # isclose's default tolerances, not exact equality.
        differs = 1.0 - lucid.isclose(s, t).float()
        terminal_term = (((terminal - terminal_target) / w_b) ** 2).mean(dim=(1, 2, 3)) * differs
        fm_term = (((velocity_fm - guided) / w_b) ** 2).mean(dim=(1, 2, 3))
        return Losses(
            loss=(terminal_term + fm_term).mean(),
            terminal=terminal_term.mean().detach(),
            flow_matching=fm_term.mean().detach(),
        )

    # ── sampling ─────────────────────────────────────────────────────────

    def generate(
        self,
        n: int,
        *,
        labels: Tensor | None = None,
        steps: int = 1,
        guidance_scale: float | None = None,
        noise: Tensor | None = None,
        use_ema: bool = True,
    ) -> Tensor:
        r"""``steps`` jumps from noise to data: :math:`x_s = x_t + (s - t)\,wF_\theta(x_t, t, s)`.

        ``[0, 1]`` is split evenly; the same network serves every step count.
        ``use_ema`` samples θ** when one is kept (what the paper evaluates).
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        c = self.config
        device = next(iter(self.net.parameters())).device.type
        if noise is None:
            noise = lucid.randn((n, c.in_channels, c.sample_size, c.sample_size), device=device)
        network = self.ema_network if use_ema and c.eval_ema_decay is not None else self.net
        x = noise
        with lucid.no_grad():
            for i in range(steps):
                t_value, s_value = 1.0 - i / steps, 1.0 - (i + 1) / steps
                t = lucid.full((n,), t_value, device=device)
                s = lucid.full((n,), s_value, device=device)
                x = x + (s_value - t_value) * network(x, t, s, labels, guidance_scale)
        return x

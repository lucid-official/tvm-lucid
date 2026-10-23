"""Terminal Velocity Matching — the equations the implementation rests on.

* the network's forward-mode stream is the derivative along ``s`` (against a
  central finite difference) and the gradient flows through it (against a
  finite difference in weight space);
* the semi-Lipschitz block is invariant to its modulation's scale, and the
  untrained field is exactly zero (adaLN-Zero);
* the scaled parameterisation multiplies by ``w``;
* the time samplers keep ``s <= t``, the terminal term vanishes at ``s = t``;
* both EMAs follow Algorithm 1 and stay out of the state dict;
* the sampler is Figure 3's jumps;
* and, where Lucid's development checkout provides
  ``lucid.models.generative.tvm``, the two implementations agree.

They need Lucid (Apple Silicon) and are skipped elsewhere.
"""

import math

import pytest

pytest.importorskip("lucid")

import lucid  # noqa: E402
import lucid.linalg  # noqa: E402
import lucid.optim  # noqa: E402
from lucid import Tensor  # noqa: E402

from tvm_lucid.tvm import TVM, TVMConfig, TVMNet  # noqa: E402


def _devices() -> list[str]:
    try:
        import lucid.metal

        return ["cpu", "metal"] if lucid.metal.is_available() else ["cpu"]
    except ImportError, AttributeError:
        return ["cpu"]


DEVICES = _devices()


def _tiny(**overrides: object) -> TVMConfig:
    base: dict[str, object] = dict(
        sample_size=8, in_channels=3, out_channels=3, patch_size=2, hidden_size=32,
        depth=2, num_heads=4, num_classes=10, frequency_embedding_size=32,
    )  # fmt: skip
    base.update(overrides)
    return TVMConfig(**base)  # type: ignore[arg-type]


def _wake(net: TVMNet, scale: float = 0.3) -> TVMNet:
    """Randomise every weight: adaLN-Zero makes the untrained field identically
    zero, so anything *about* the conditioning has to break that first."""
    with lucid.no_grad():
        for param in net.parameters():
            param.copy_(lucid.randn(tuple(param.shape), device=param.device.type) * scale)
    return net


def _inputs(device: str, batch: int = 3) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = lucid.randn((batch, 3, 8, 8), device=device)
    t = lucid.tensor([0.9, 0.6, 0.35][:batch], device=device)
    s = lucid.tensor([0.2, 0.5, 0.3][:batch], device=device)
    labels = lucid.tensor([1, 4, 10][:batch], device=device)
    w = lucid.tensor([2.0, 1.5, 1.0][:batch], device=device)
    return x, t, s, labels, w


def _max(a: Tensor) -> float:
    return float(a.abs().max().item())


# ── the network and its derivative along s ───────────────────────────────


class TestNetwork:
    def test_the_untrained_field_is_exactly_zero(self) -> None:
        net = TVMNet(_tiny()).eval()
        x, t, s, labels, w = _inputs("cpu")
        assert _max(net(x, t, s, labels, w)) == 0.0

    def test_linear_layers_start_at_unit_spectral_norm(self) -> None:
        net = TVMNet(_tiny())
        for weight in (net.blocks[0].attn.qkv.weight, net.blocks[1].mlp.fc2.weight):
            sigma = float(lucid.linalg.svdvals(weight)[0].item())
            assert math.isclose(sigma, 1.0, rel_tol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_jvp_primal_is_the_forward(self, device: str) -> None:
        lucid.manual_seed(0)
        net = _wake(TVMNet(_tiny()).to(device)).eval()
        x, t, s, labels, w = _inputs(device)
        with lucid.no_grad():
            out, _ = net.jvp_s(x, t, s, labels, w)
            ref = net(x, t, s, labels, w)
        assert _max(out - ref) <= 1e-4 * _max(ref)

    @pytest.mark.parametrize("device", DEVICES)
    def test_tangent_is_the_derivative_along_s(self, device: str) -> None:
        lucid.manual_seed(1)
        net = _wake(TVMNet(_tiny()).to(device)).eval()
        x, t, s, labels, w = _inputs(device)
        h = 3e-3
        with lucid.no_grad():
            _, along_s = net.jvp_s(x, t, s, labels, w)
            fd = (net(x, t, s + h, labels, w) - net(x, t, s - h, labels, w)) / (2 * h)
        assert _max(fd) > 1e-2  # the check is not vacuous
        assert _max(along_s - fd) <= 2e-3 * _max(fd) + 1e-4

    def test_the_gradient_flows_through_the_tangent(self) -> None:
        """d/dθ ⟨∂u/∂s, R⟩ against a central difference in weight space.

        A plain central difference is off by O(h²) — several percent at
        h = 1e-2 for a random direction through a deep net — so two step sizes
        are combined (Richardson) to cancel that term; what remains is ~1e-4.
        """
        lucid.manual_seed(2)
        net = _wake(TVMNet(_tiny())).eval()
        x, t, s, labels, w = _inputs("cpu")
        probe = lucid.randn((3, 3, 8, 8))
        weight = net.blocks[0].attn.qkv.weight
        direction = lucid.randn(tuple(weight.shape))

        _, along_s = net.jvp_s(x, t, s, labels, w)
        (along_s * probe).sum().backward()
        analytic = float((weight.grad * direction).sum().item())

        original = weight.detach().clone()

        def objective(shift: float) -> float:
            with lucid.no_grad():
                weight.copy_(original + direction * shift)
                return float((net.jvp_s(x, t, s, labels, w)[1] * probe).sum().item())

        def central(h: float) -> float:
            return (objective(h) - objective(-h)) / (2 * h)

        numeric = (4.0 * central(5e-3) - central(1e-2)) / 3.0
        with lucid.no_grad():
            weight.copy_(original)
        assert abs(numeric) > 1e-2
        assert abs(analytic - numeric) <= 5e-3 * abs(numeric)

    def test_gradient_checkpointing_changes_no_gradient(self) -> None:
        """Recomputing each block in backward gives the gradients of keeping it."""
        lucid.manual_seed(13)
        plain = _wake(TVMNet(_tiny())).train()
        recomputed = TVMNet(_tiny(gradient_checkpointing=True)).train()
        recomputed.load_state_dict(plain.state_dict())
        x, t, s, labels, w = _inputs("cpu")
        probe = lucid.randn((3, 3, 8, 8))
        for net in (plain, recomputed):
            u, du = net.jvp_s(x, t, s, labels, w)
            ((u + du) * probe).sum().backward()
            (net(x, s, s, labels, w) * probe).sum().backward()
        reference = dict(plain.named_parameters())
        for name, param in recomputed.named_parameters():
            assert param.grad is not None, name
            expected = reference[name].grad
            assert _max(param.grad - expected) <= 1e-4 * max(_max(expected), 1e-6), name

    def test_the_block_is_invariant_to_its_modulation_scale(self) -> None:
        """Eq. 16: each shift, scale and gate is RMS-normalised, so scaling the
        modulation projection (weight and bias) changes nothing."""
        lucid.manual_seed(3)
        net = _wake(TVMNet(_tiny())).eval()
        x, t, s, labels, w = _inputs("cpu")
        with lucid.no_grad():
            before = net(x, t, s, labels, w)
            for linear in [block.ada_ln[1] for block in net.blocks] + [net.final.ada_ln[1]]:
                linear.weight.copy_(linear.weight * 7.0)
                linear.bias.copy_(linear.bias * 7.0)
            after = net(x, t, s, labels, w)
        assert _max(after - before) <= 1e-4 * _max(before)

    def test_without_the_modulation_norm_the_scale_matters(self) -> None:
        lucid.manual_seed(3)
        net = _wake(TVMNet(_tiny(modulation_norm=False))).eval()
        x, t, s, labels, w = _inputs("cpu")
        with lucid.no_grad():
            before = net(x, t, s, labels, w)
            linear = net.final.ada_ln[1]
            linear.weight.copy_(linear.weight * 7.0)
            after = net(x, t, s, labels, w)
        assert _max(after - before) > 1e-2 * _max(before)

    def test_the_scaled_parameterisation_multiplies_by_w(self) -> None:
        lucid.manual_seed(4)
        scaled = _wake(TVMNet(_tiny())).eval()
        plain = TVMNet(_tiny(scaled_parameterization=False)).eval()
        plain.load_state_dict(scaled.state_dict())
        x, t, s, labels, w = _inputs("cpu")
        with lucid.no_grad():
            gap = scaled(x, t, s, labels, w) - plain(x, t, s, labels, w) * w.reshape(-1, 1, 1, 1)
        assert _max(gap) <= 1e-5

    def test_no_label_means_the_null_label_at_unit_weight(self) -> None:
        lucid.manual_seed(5)
        net = _wake(TVMNet(_tiny())).eval()
        x, t, s, _, _ = _inputs("cpu")
        null = lucid.full((3,), 10, dtype=lucid.int64)
        with lucid.no_grad():
            gap = net(x, t, s) - net(x, t, s, null, lucid.ones((3,)))
        assert _max(gap) == 0.0


# ── the objective ────────────────────────────────────────────────────────


class TestObjective:
    @pytest.mark.parametrize("sampler", ["gap", "trunc", "clamp"])
    def test_time_samplers_keep_s_below_t(self, sampler: str) -> None:
        lucid.manual_seed(6)
        model = TVM(_tiny(time_sampler=sampler))
        t, s, s_fm = model.sample_times(20_000, "cpu")
        assert float((s - t).max().item()) <= 1e-6
        for times in (t, s, s_fm):
            assert 0.0 <= float(times.min().item()) and float(times.max().item()) <= 1.0

    def test_random_guidance_draws_the_reciprocal_uniformly(self) -> None:
        lucid.manual_seed(7)
        model = TVM(_tiny(guidance_range=(1.5, 3.0)))
        beta = 1.0 / model.sample_guidance(20_000, "cpu")
        assert float(beta.min().item()) >= 1.0 / 3.0 - 1e-6
        assert float(beta.max().item()) <= 1.0 / 1.5 + 1e-6
        assert abs(float(beta.mean().item()) - 0.5 * (1.0 / 1.5 + 1.0 / 3.0)) < 5e-3

    def test_the_terminal_term_vanishes_when_s_equals_t(self) -> None:
        lucid.manual_seed(8)
        model = TVM(_tiny(ratio_t_equals_s=1.0))
        _wake(model.net)
        out = model(lucid.randn((4, 3, 8, 8)).clamp(-1.0, 1.0), lucid.tensor([0, 1, 2, 3]))
        assert float(out.terminal.item()) == 0.0
        assert float(out.flow_matching.item()) > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("detach_jvp", [False, True])
    def test_a_training_step_is_finite_and_moves_the_weights(
        self, device: str, detach_jvp: bool
    ) -> None:
        lucid.manual_seed(9)
        model = TVM(_tiny(detach_jvp=detach_jvp)).to(device).train()
        opt = lucid.optim.AdamW(model.parameters(), lr=1e-3)
        images = lucid.randn((4, 3, 8, 8), device=device).clamp(-1.0, 1.0)
        labels = lucid.tensor([0, 1, 2, 3], device=device)
        before = model.net.final.proj.weight.detach().clone()
        for _ in range(2):  # the second step sees a non-zero field
            out = model(images, labels)
            assert math.isfinite(float(out.loss.item()))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            model.update_ema()
        assert _max(model.net.final.proj.weight.detach() - before) > 0.0


# ── Algorithm 1's two averages ───────────────────────────────────────────


class TestEMA:
    def test_averages_start_at_the_live_weights_and_stay_out_of_the_state_dict(self) -> None:
        model = TVM(_tiny())
        _wake(model.net)
        live = dict(model.net.named_parameters())
        for average in (model.target_network, model.ema_network):
            for name, param in average.named_parameters():
                assert not param.requires_grad
                assert _max(param - live[name]) == 0.0
        assert set(model.state_dict()) == {f"net.{k}" for k in model.net.state_dict()}
        assert sum(1 for _ in model.parameters()) == sum(1 for _ in model.net.parameters())

    def test_update_follows_algorithm_1(self) -> None:
        model = TVM(_tiny(target_ema_decay=0.9, eval_ema_decay=0.99))
        name = "blocks.0.attn.qkv.weight"
        target0 = dict(model.target_network.named_parameters())[name].detach().clone()
        eval0 = dict(model.ema_network.named_parameters())[name].detach().clone()
        _wake(model.net)
        live = dict(model.net.named_parameters())[name].detach()
        model.update_ema()
        target1 = dict(model.target_network.named_parameters())[name]
        eval1 = dict(model.ema_network.named_parameters())[name]
        assert _max(target1 - (target0 * 0.9 + live * 0.1)) <= 1e-6
        assert _max(eval1 - (eval0 * 0.99 + live * 0.01)) <= 1e-6

    def test_a_zero_target_rate_reads_the_live_network(self) -> None:
        model = TVM(_tiny(target_ema_decay=0.0))
        assert model.target_network is model.net


# ── the sampler (Figure 3) ───────────────────────────────────────────────


class TestSampler:
    def test_steps_are_even_jumps_along_the_average_velocity(self) -> None:
        lucid.manual_seed(10)
        model = TVM(_tiny(eval_ema_decay=None))
        net = _wake(model.net).eval()
        noise = lucid.randn((2, 3, 8, 8))
        labels = lucid.tensor([3, 7])
        ones, half, zeros = (lucid.full((2,), v) for v in (1.0, 0.5, 0.0))
        with lucid.no_grad():
            one = model.generate(2, labels=labels, steps=1, noise=noise)
            one_ref = noise - net(noise, ones, zeros, labels)
            two = model.generate(2, labels=labels, steps=2, noise=noise)
            mid = noise - 0.5 * net(noise, ones, half, labels)
            two_ref = mid - 0.5 * net(mid, half, zeros, labels)
        assert _max(one - one_ref) <= 1e-6
        assert _max(two - two_ref) <= 1e-6

    def test_the_trajectory_runs_from_the_noise_to_the_sample(self) -> None:
        lucid.manual_seed(14)
        model = TVM(_tiny(eval_ema_decay=None))
        _wake(model.net).eval()
        noise = lucid.randn((2, 3, 8, 8))
        labels = lucid.tensor([3, 7])
        with lucid.no_grad():
            path = model.trajectory(2, labels=labels, steps=4, noise=noise)
            final = model.generate(2, labels=labels, steps=4, noise=noise)
        assert len(path) == 5
        assert _max(path[0] - noise) == 0.0
        assert _max(path[-1] - final) == 0.0


# ── agreement with Lucid's model-zoo implementation, where it exists ────


class TestZooParity:
    """Runs only against a Lucid checkout that ships ``lucid.models.generative.tvm``
    (development ``main`` after 8739862); the PyPI release skips it."""

    @pytest.fixture
    def zoo(self):  # type: ignore[no-untyped-def]
        return pytest.importorskip("lucid.models.generative.tvm")

    @staticmethod
    def _zoo_config(zoo, **overrides: object):  # type: ignore[no-untyped-def]
        fields = dict(_tiny(**overrides).__dict__)
        fields.pop("gradient_checkpointing")  # engineering, not a field of the method
        return zoo.TVMConfig(**fields, latent_mean=(0.0,) * 3, latent_std=(1.0,) * 3)

    def test_the_networks_agree(self, zoo) -> None:  # type: ignore[no-untyped-def]
        lucid.manual_seed(11)
        mine = _wake(TVMNet(_tiny())).eval()
        theirs = zoo.TVMModel(self._zoo_config(zoo)).eval()
        theirs.load_state_dict(mine.state_dict())
        x, t, s, labels, w = _inputs("cpu")
        with lucid.no_grad():
            ref = theirs(x, t, s, labels, w).sample
            gap = mine(x, t, s, labels, w) - ref
        assert _max(gap) <= 1e-5 * max(_max(ref), 1.0)

    def test_the_losses_and_gradients_agree(self, zoo) -> None:  # type: ignore[no-untyped-def]
        lucid.manual_seed(12)
        mine = TVM(_tiny(class_dropout=0.3))
        _wake(mine.net)
        theirs = zoo.TVMForImageGeneration(self._zoo_config(zoo, class_dropout=0.3))
        theirs.tvm.load_state_dict(mine.net.state_dict())
        images = lucid.randn((4, 3, 8, 8)).clamp(-1.0, 1.0)
        labels = lucid.tensor([0, 1, 2, 3])

        lucid.manual_seed(13)
        a = mine(images, labels)
        a.loss.backward()
        lucid.manual_seed(13)
        b = theirs(images, labels)
        b.loss.backward()
        assert math.isclose(float(a.loss.item()), float(b.loss.item()), rel_tol=1e-4)
        theirs_grads = dict(theirs.tvm.named_parameters())
        for name, param in mine.net.named_parameters():
            other = theirs_grads[name].grad
            assert other is not None and param.grad is not None
            assert _max(param.grad - other) <= 1e-3 * max(_max(other), 1e-6), name

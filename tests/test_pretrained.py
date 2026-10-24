"""The released checkpoints' names land on TVMNet one-to-one.

``scripts/sample_released.py`` and the parity numbers in the README need
the real files; this only pins the mapping, which a renamed parameter here
would otherwise break silently until someone loads a release.
"""

from tvm_lucid import pretrained
from tvm_lucid.tvm import TVMConfig, TVMNet


def _upstream_names(depth: int) -> list[str]:
    """``TVMWrapper.state_dict()`` of upstream's DiT, as the converter writes it."""
    pairs = ("weight", "bias")
    names = [f"model.x_embedder.proj.{p}" for p in pairs] + ["model.pos_embed"]
    for embedder in ("t_embedder", "s_embedder", "guidance_embedder"):
        names += [f"model.{embedder}.mlp.{i}.{p}" for i in (0, 2) for p in pairs]
    names.append("model.y_embedder.embedding_table.weight")
    for b in range(depth):
        block = f"model.blocks.{b}"
        names += [f"{block}.attn.qkv.{p}" for p in pairs]
        names += [f"{block}.attn.q_norm.weight", f"{block}.attn.k_norm.weight"]
        names += [f"{block}.attn.proj.{p}" for p in pairs]
        names += [f"{block}.mlp.fc{i}.{p}" for i in (1, 2) for p in pairs]
        names += [f"{block}.adaLN_modulation.1.{p}" for p in pairs]
    names += [f"model.final_layer.linear.{p}" for p in pairs]
    names += [f"model.final_layer.adaLN_modulation.1.{p}" for p in pairs]
    return names


def test_rename_is_a_bijection_onto_tvmnet() -> None:
    config = TVMConfig(sample_size=8, hidden_size=64, depth=2, num_heads=4)
    upstream = [n for n in _upstream_names(config.depth) if n != "model.pos_embed"]
    renamed = [pretrained.rename(n) for n in upstream]
    assert len(set(renamed)) == len(renamed)
    assert set(renamed) == set(TVMNet(config).state_dict())


def test_release_configs_are_xl2() -> None:
    for release in pretrained.RELEASES.values():
        config = release.config()
        assert (config.hidden_size, config.depth, config.num_heads) == (1152, 28, 18)
        assert config.sample_size * 8 == release.image_size

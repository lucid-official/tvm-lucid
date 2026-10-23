"""Terminal Velocity Matching (Zhou, Parger, Haque & Song, ICLR 2026), written
against Lucid's public API only.

    config.py   TVMConfig — every choice the paper makes, with its default
    layers.py   the semi-Lipschitz DiT's layers, each with its derivative along s
    network.py  TVMNet — the two-time velocity network
    method.py   TVM — the objective (Eq. 15), Algorithm 1's EMAs, the sampler
"""

from tvm_lucid.tvm.config import TVMConfig
from tvm_lucid.tvm.method import TVM, Losses
from tvm_lucid.tvm.network import TVMNet

__all__ = ["TVM", "TVMConfig", "TVMNet", "Losses"]

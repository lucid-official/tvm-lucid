"""The project is written against Lucid's public API, as a user of the library.

Nothing under ``tvm_lucid/`` may reach into Lucid's private modules
(``lucid._*``, ``lucid.models._*``) or lean on its model-zoo TVM: the method
is implemented here.  The one zoo model used is the pretrained Inception-v3
classifier that feeds FID, as any user would.
"""

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "tvm_lucid"

# ``lucid._x`` / ``lucid.models._x`` — but not dunders such as ``lucid.__version__``.
_PRIVATE = re.compile(r"\blucid(\.\w+)*\._(?!_)\w+")
# Importing or calling the zoo's TVM (prose that mentions it is fine).
_ZOO_TVM = re.compile(
    r"^\s*(from|import)\s+lucid\.models\.generative"
    r"|\b(M|models)\.(tvm_\w+|TVM\w*)\b"
    r"|\bTVMForImageGeneration\("
)


def _sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_private_lucid_modules() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{n}: {line.strip()}"
        for path in _sources()
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _PRIVATE.search(line)
    ]
    assert not offenders, "\n".join(offenders)


def test_the_method_is_not_borrowed_from_the_model_zoo() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{n}: {line.strip()}"
        for path in _sources()
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _ZOO_TVM.search(line)
    ]
    assert not offenders, "\n".join(offenders)

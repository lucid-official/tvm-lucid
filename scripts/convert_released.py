"""Convert a released TVM checkpoint (``lumaai/tvm`` on Hugging Face) to SafeTensors.

    python -I scripts/convert_released.py ~/.cache/tvm_lucid/download/lumaai-tvm/im512_w2p5.pkl \
        ~/.cache/tvm_lucid/im512_w2p5.safetensors

The release ships EDM-style pickles: ``torch_utils.persistence`` stores each
network class's *source code* in the file and ``exec``s it on load, so
``pickle.load`` on one runs whatever the file says.  This converter never
does.  Every global the pickle names resolves to an inert stub, except the
handful that rebuild tensors, and those only slice bytes out of the file
with numpy.  No torch is needed.

Only the evaluated network is written: ``net``, which upstream's snapshots
fill with the evaluation EMA (``ema`` where a file has one).  It goes out
under the upstream parameter names (``model.blocks.0.attn.qkv.weight`` …),
in the stored dtype.  ``tvm_lucid.pretrained`` maps those names onto
:class:`tvm_lucid.tvm.TVMNet`.
"""

import argparse
import collections
import io
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

# Storage class → (numpy dtype, SafeTensors code).  bfloat16 has no numpy
# dtype; its bits are carried as uint16 and labelled BF16.
_STORAGES = {
    "DoubleStorage": (np.float64, "F64"),
    "FloatStorage": (np.float32, "F32"),
    "HalfStorage": (np.float16, "F16"),
    "BFloat16Storage": (np.uint16, "BF16"),
    "LongStorage": (np.int64, "I64"),
    "IntStorage": (np.int32, "I32"),
    "ShortStorage": (np.int16, "I16"),
    "CharStorage": (np.int8, "I8"),
    "ByteStorage": (np.uint8, "U8"),
    "BoolStorage": (np.bool_, "BOOL"),
}
_TORCH_DTYPES = {  # the dtype objects a zip-format checkpoint names
    "float64": "DoubleStorage", "float32": "FloatStorage", "float16": "HalfStorage",
    "bfloat16": "BFloat16Storage", "int64": "LongStorage", "int32": "IntStorage",
    "int16": "ShortStorage", "int8": "CharStorage", "uint8": "ByteStorage", "bool": "BoolStorage",
}  # fmt: skip


class Stub(dict):
    """Whatever the pickle builds that is not a tensor: it records, never runs.

    A dict so ``SETITEMS`` works, with ``append``/``extend`` for ``APPENDS``;
    constructor arguments and ``BUILD`` state are kept for inspection.
    """

    qualname = "?"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args, self.kwargs, self.state, self.items_ = args, kwargs, None, []

    def __new__(cls, *args: Any, **kwargs: Any) -> Stub:
        obj = super().__new__(cls)
        obj.args, obj.kwargs, obj.state, obj.items_ = args, kwargs, None, []
        return obj

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def append(self, item: Any) -> None:
        self.items_.append(item)

    def extend(self, items: Any) -> None:
        self.items_.extend(items)

    def __reduce__(self) -> Any:  # never re-pickled; keeps copy/pickle honest
        raise TypeError("a stub from an untrusted pickle")


def _stub_class(module: str, name: str) -> type:
    return type(name, (Stub,), {"qualname": f"{module}.{name}"})


class Persistent(Stub):
    """``torch_utils.persistence._reconstruct_persistent_obj(meta)``, not executed."""

    qualname = "torch_utils.persistence"


class StorageType:
    def __init__(self, name: str, label: str | None = None) -> None:
        self.name, self.label = name, label or name


def _strided(storage: np.ndarray, offset: int, size: tuple, stride: tuple) -> np.ndarray:
    item = storage.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)
    )
    return np.ascontiguousarray(view)


def _rebuild_tensor(storage: Any, offset: int, size: tuple, stride: tuple, *_: Any) -> np.ndarray:
    if not isinstance(storage, np.ndarray):
        raise TypeError(f"tensor storage is {type(storage).__name__}, not bytes")
    return _strided(storage, offset, size, stride)


def _rebuild_parameter(data: Any, *_: Any) -> Any:
    return data


class _Restricted(pickle.Unpickler):
    """Resolves only the names that rebuild tensors; everything else is a stub."""

    def __init__(self, file: Any, storages: Any = None) -> None:
        super().__init__(file)
        self._storages = storages

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch._utils" and name in (
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
        ):
            return _rebuild_parameter
        if module == "torch.storage" and name == "_load_from_bytes":
            return _load_legacy_storage
        if module == "torch" and name in _STORAGES:
            return StorageType(name)
        if module == "torch" and name in _TORCH_DTYPES:
            return StorageType(_TORCH_DTYPES[name], label=name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch_utils.persistence" and name == "_reconstruct_persistent_obj":
            return Persistent
        if module == "builtins" and name in ("set", "frozenset", "slice", "complex"):
            return {"set": set, "frozenset": frozenset, "slice": slice, "complex": complex}[name]
        return _stub_class(module, name)

    def persistent_load(self, pid: Any) -> Any:
        if self._storages is None:
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        return self._storages(pid)


def _load_legacy_storage(blob: bytes) -> np.ndarray:
    """``torch.storage._load_from_bytes``: one storage in the legacy (non-zip) format.

    Four pickles — magic number, protocol version, system info, then the
    storage as a persistent id ``('storage', type, key, location, numel, view)``
    — a fifth with the storage keys, then per key an int64 element count and
    the raw little-endian elements.
    """
    f = io.BytesIO(blob)
    for _ in range(3):
        _Restricted(f).load()
    ids: list[tuple] = []

    def record(pid: Any) -> tuple:
        ids.append(tuple(pid))
        return tuple(pid)

    root = _Restricted(f, storages=record).load()
    keys = _Restricted(f).load()
    arrays = {}
    for key in keys:
        (_, kind, key_, _location, _numel, *_rest) = next(p for p in ids if p[2] == key)
        dtype = np.dtype(_STORAGES[kind.name][0]).newbyteorder("<")
        (count,) = struct.unpack("<q", f.read(8))
        arrays[key_] = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    if not isinstance(root, tuple) or root[0] != "storage":
        raise pickle.UnpicklingError("a storage blob that holds no storage")
    return arrays[root[2]]


def _load(path: Path) -> Any:
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":  # a zip-format ``torch.save`` file
        archive = zipfile.ZipFile(path)
        (pkl,) = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        prefix = pkl[: -len("data.pkl")]

        def storage(pid: Any) -> np.ndarray:
            _, kind, key, _location, numel = pid[:5]
            dtype = np.dtype(_STORAGES[kind.name][0]).newbyteorder("<")
            raw = archive.read(f"{prefix}data/{key}")
            return np.frombuffer(raw, dtype=dtype, count=numel)

        return _Restricted(io.BytesIO(archive.read(pkl)), storages=storage).load()
    with path.open("rb") as fh:
        return _Restricted(fh).load()


# ── the module tree ──────────────────────────────────────────────────────


def _module_dict(obj: Any) -> dict | None:
    """The ``__dict__`` a module was pickled with, if ``obj`` is one."""
    if isinstance(obj, Persistent):
        meta = obj.args[0] if obj.args else {}
        state = meta.get("state") if isinstance(meta, dict) else None
    elif isinstance(obj, Stub):
        state = obj.state
        if isinstance(state, tuple):  # (dict, slots)
            state = state[0]
    else:
        return None
    if isinstance(state, dict) and "_parameters" in state:
        return state
    return None


def unwrap(obj: Any) -> Any:
    """The module inside a ``torch.compile`` wrapper.

    ``OptimizedModule`` pickles as ``(cls, (module, context))`` — the form the
    releases take — so the module is its first constructor argument.
    """
    if _module_dict(obj) is None and isinstance(obj, Stub) and obj.args:
        if _module_dict(obj.args[0]) is not None:
            return obj.args[0]
    return obj


def flatten(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """``state_dict()`` of a stubbed module: parameters and persistent buffers."""
    state = _module_dict(obj)
    if state is None:
        raise TypeError(f"{prefix or 'root'} is not a module")
    out: dict[str, np.ndarray] = {}
    skipped = set(state.get("_non_persistent_buffers_set") or ())
    for group in ("_parameters", "_buffers"):
        for name, value in (state.get(group) or {}).items():
            if value is None or name in skipped:
                continue
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{prefix}{name} is {type(value).__name__}, not a tensor")
            out[prefix + name] = value
    for name, child in (state.get("_modules") or {}).items():
        if child is not None and _module_dict(child) is not None:
            out.update(flatten(child, f"{prefix}{name}."))
    return out


def _describe(obj: Any) -> str:
    if isinstance(obj, Persistent):
        meta = obj.args[0]
        return f"persistent {meta.get('class_name')}"
    if isinstance(obj, Stub):
        return obj.qualname
    return type(obj).__name__


def _plain(value: Any) -> Any:
    """JSON-able settings (``img_resolution``, ``do_scale_param`` …) for the header."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, StorageType):
        return value.label
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict) and not isinstance(value, Stub):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Stub) and value.qualname.startswith("omegaconf"):
        return "<omegaconf>"
    return f"<{_describe(value)}>"


def settings(obj: Any) -> dict[str, Any]:
    """The non-tensor attributes of the top module (resolution, flags…)."""
    state = _module_dict(obj) or {}
    return {
        k: _plain(v)
        for k, v in state.items()
        if not k.startswith("_") and k not in ("training",) and not isinstance(v, np.ndarray)
    }


def write_safetensors(tensors: dict[str, np.ndarray], codes: dict[str, str], path: Path,
                      metadata: dict[str, str]) -> None:  # fmt: skip
    header: dict[str, Any] = {}
    offset = 0
    for name, array in tensors.items():
        header[name] = {
            "dtype": codes[name],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for array in tensors.values():
            fh.write(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pickle", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--key", default=None, help="top-level entry (default: ema, else net)")
    args = parser.parse_args()

    data = _load(args.pickle)
    if not isinstance(data, dict):
        sys.exit(f"top level is {_describe(data)}, expected a dict")
    print("top-level entries:", {k: _describe(v) for k, v in data.items()})
    key = args.key or ("ema" if "ema" in data else "net")
    root = unwrap(data[key])
    tensors = flatten(root)
    if all(name.startswith("_orig_mod.") for name in tensors):  # a compiled module
        tensors = {name[len("_orig_mod.") :]: v for name, v in tensors.items()}

    codes = {}
    for name, array in tensors.items():
        kind = next(k for k, (dt, _) in _STORAGES.items() if np.dtype(dt) == array.dtype)
        codes[name] = _STORAGES[kind][1]
    count = sum(int(a.size) for a in tensors.values())
    attrs = settings(root)
    print(f"{key}: {_describe(root)}, {len(tensors)} tensors, {count:,} elements")
    print("settings:", json.dumps(attrs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": args.pickle.name,
        "entry": key,
        "settings": json.dumps(attrs),
        "format": "upstream parameter names (github.com/lumalabs/tvm)",
    }
    write_safetensors(tensors, codes, args.out, metadata)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()

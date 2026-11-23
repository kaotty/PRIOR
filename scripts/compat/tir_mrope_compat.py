"""Opt-in PyTorch 2.9 jagged mRoPE layout repair, outside veRL sources.

Equal-length (4, L) samples make as_nested_tensor infer the channel axis
as ragged. FSDP packing requires the sequence axis to be ragged instead.
Enable at Python startup through sitecustomize.py and
TIR_MROPE_LAYOUT_COMPAT=1. No tokens or position values are changed.
"""

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "verl.utils.tensordict_utils"


def normalize_positions(data):
    if data is None or "position_ids" not in data.keys():
        return data
    import torch

    pos = data["position_ids"]
    if not isinstance(pos, torch.Tensor) or not pos.is_nested or pos.dim() != 3:
        return data
    ids = data.get("input_ids")
    if not isinstance(ids, torch.Tensor) or not ids.is_nested or ids.dim() != 2:
        raise RuntimeError("mRoPE layout repair requires packed input_ids lengths")
    values, offsets = pos.values(), ids.offsets()
    batch_size = len(data)
    total = ids.values().numel()
    if values.ndim != 2 or values.numel() != 4 * total:
        raise RuntimeError(f"Unexpected Qwen mRoPE values shape: {tuple(values.shape)}, tokens={total}")
    canonical = tuple(values.shape) == (4, total) and torch.equal(pos.offsets(), offsets)
    if canonical:
        if pos._ragged_idx == 2:
            return data
    else:
        length = total // batch_size
        equal_lengths = total == batch_size * length and bool(torch.all(offsets.diff() == length))
        if not (equal_lengths and tuple(values.shape) == (4 * batch_size, length)
                and bool(torch.all(pos.offsets().diff() == 4))):
            raise RuntimeError(f"Unrecognized mRoPE layout: values={tuple(values.shape)}, batch={batch_size}")
        values = values.reshape(batch_size, 4, length).permute(1, 0, 2).reshape(4, total)
        if not getattr(normalize_positions, "reported", False):
            print(f"[tir-mrope-compat] repaired equal-length layout: batch={batch_size}, length={length}", flush=True)
            normalize_positions.reported = True
    data["position_ids"] = torch.nested.nested_tensor_from_jagged(
        values, offsets=offsets, jagged_dim=2
    )
    return data


def patch_module(module):
    if getattr(module, "_tir_mrope_compat", False):
        return
    original_select = module.index_select_tensor_dict
    original_chunk = module.chunk_tensordict

    def index_select_tensor_dict(batch, indices):
        return normalize_positions(original_select(normalize_positions(batch), indices))

    def chunk_tensordict(td, chunks):
        return [normalize_positions(part) for part in original_chunk(normalize_positions(td), chunks)]

    def maybe_fix_3d_position_ids(data):
        normalize_positions(data)

    module.index_select_tensor_dict = index_select_tensor_dict
    module.chunk_tensordict = chunk_tensordict
    module.maybe_fix_3d_position_ids = maybe_fix_3d_position_ids
    module._tir_mrope_compat = True
    print("[tir-mrope-compat] explicit sequence-axis layout enabled", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        patch_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if os.environ.get("TIR_MROPE_LAYOUT_COMPAT") != "1":
        return
    if TARGET in sys.modules:
        patch_module(sys.modules[TARGET])
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

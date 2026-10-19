"""
Load Hugging Face safetensors weights into an NNX module.
"""
from __future__ import annotations

import glob
import os
from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file


def iter_named_params(module, prefix: str = ""):
    """Yield ``(dotted_path, nnx.Param)`` for every parameter of an NNX module tree."""
    for name, val in vars(module).items():
        if name.startswith("_"):
            continue
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(val, nnx.Param):
            yield path, val
        elif isinstance(val, (list, tuple, nnx.List)):
            for i, item in enumerate(val):
                if isinstance(item, nnx.Module):
                    yield from iter_named_params(item, f"{path}.{i}")
        elif isinstance(val, nnx.Module):
            yield from iter_named_params(val, path)


def snapshot(model_name: str, revision: str | None = None) -> str:
    """A local directory holding the model: a path, or the (cached) HF hub snapshot."""
    local_path = os.path.abspath(os.path.expanduser(model_name))
    if os.path.isdir(local_path):
        return local_path
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model_name, revision=revision, local_files_only=True)
    except Exception:
        return snapshot_download(model_name, revision=revision)


def load_snapshot_tensors(directory: str) -> dict[str, np.ndarray]:
    """Every tensor of every ``*.safetensors`` shard in an HF snapshot directory."""
    tensors = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        tensors.update(load_file(path))
    if not tensors:
        raise FileNotFoundError(f"no safetensors files under {directory}")
    return tensors


def load_hf_state(model, tensors: dict[str, np.ndarray], dtype, *, ignore: Iterable[str] = ()) -> None:
    """Copy ``tensors`` into ``model`` and assert that every tensor and every Param matched."""
    params = dict(iter_named_params(model))
    ignored = set(ignore)
    filled, consumed = set(), set()
    for tname, value in tensors.items():
        if tname in ignored:
            consumed.add(tname)
            continue
        jname = tname
        if tname not in params and tname.endswith(".weight"):
            jname = tname[: -len(".weight")] + ".kernel"
        if jname not in params:
            continue
        param = params[jname]
        if jname != tname:
            assert value.ndim == 2, f"{tname}: kernel mapping expects 2-D, got {value.shape}"
            value = value.T
        assert tuple(value.shape) == tuple(param[...].shape), \
            f"{tname}: checkpoint {value.shape} != param {param[...].shape}"
        param.set_value(jnp.asarray(np.asarray(value), dtype=dtype))
        consumed.add(tname)
        filled.add(jname)
    unconsumed = sorted(set(tensors) - consumed)
    unfilled = sorted(set(params) - filled)
    assert not unconsumed and not unfilled, (
        f"checkpoint<->model bijection failed: unconsumed={unconsumed[:5]} unfilled={unfilled[:5]}")

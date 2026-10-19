"""
Checkpoints as flat safetensors files plus a JSON sidecar.
"""
from __future__ import annotations

import json
import os
import shutil

import jax
import numpy as np
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes


# ------------------------------------------------------------ pytree <-> flat dict
def _key(path) -> str:
    """``['blocks'][0]['attn']['qkv']['kernel'].value`` -> ``blocks.0.attn.qkv.kernel``."""
    parts = []
    for entry in path:
        if isinstance(entry, jax.tree_util.GetAttrKey):
            parts.append(entry.name)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            parts.append(str(entry.idx))
        else:
            parts.append(str(getattr(entry, "key", entry)))
    if parts and parts[-1] == "value":  # the array inside an nnx.Variable
        parts.pop()
    return ".".join(parts)


def flatten(tree) -> dict[str, np.ndarray]:
    """Flatten a pytree (an ``nnx.State``, an optax state, ...) into ``{path: array}``."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(leaf)
        out[_key(path)] = array if array.ndim == 0 else np.ascontiguousarray(array)
    return out


def unflatten(template, arrays: dict[str, np.ndarray]):
    """Rebuild a pytree shaped like ``template`` from :func:`flatten`'s dictionary."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    keys = list(flatten(template))
    missing = [k for k in keys if k not in arrays]
    if missing:
        raise KeyError(f"checkpoint is missing {len(missing)} arrays, e.g. {missing[:3]}")
    values = []
    for key, leaf in zip(keys, leaves):
        value = arrays[key]
        if tuple(value.shape) != tuple(leaf.shape):
            raise ValueError(f"{key}: checkpoint shape {value.shape} != expected {leaf.shape}")
        values.append(value.astype(leaf.dtype))
    return jax.tree_util.tree_unflatten(treedef, values)


# ------------------------------------------------------------------------- files
def write_bytes(path: str, data: bytes) -> None:
    """Atomic local write (temporary file + rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


# ------------------------------------------------------------------- save / load
def save(directory: str, trees: dict, meta: dict) -> None:
    """Write ``{name: pytree}`` as ``<directory>/<name>.safetensors`` plus ``meta.json``.

    Every host holds the same replicated arrays, so only process 0 writes.
    """
    for tree in trees.values():
        jax.block_until_ready(tree)
    if jax.process_index() != 0:
        return
    for name, tree in trees.items():
        write_bytes(f"{directory}/{name}.safetensors", save_safetensors_bytes(flatten(jax.device_get(tree))))
    write_bytes(f"{directory}/meta.json", json.dumps(meta, indent=2, sort_keys=True).encode())


def load_meta(directory: str) -> dict:
    return json.loads(read_bytes(f"{directory}/meta.json"))


def load_arrays(directory: str, name: str) -> dict[str, np.ndarray]:
    return load_safetensors_bytes(read_bytes(f"{directory}/{name}.safetensors"))


def load(directory: str, name: str, template):
    """Restore one named pytree, shaped and typed like ``template``."""
    return unflatten(template, load_arrays(directory, name))


# ------------------------------------------------------------ rolling checkpoints
def latest_dir(run_dir: str) -> str | None:
    """The newest complete rolling checkpoint of a run, or ``None``."""
    pointer = f"{run_dir}/ckpt/LATEST"
    if not os.path.exists(pointer):
        return None
    return f"{run_dir}/ckpt/{json.loads(read_bytes(pointer))['current']}"


def save_rolling(run_dir: str, step: int, trees: dict, meta: dict) -> str:
    """Save a resumable checkpoint, update ``LATEST`` last, and prune to the newest two."""
    name = f"step-{step:09d}"
    directory = f"{run_dir}/ckpt/{name}"
    save(directory, trees, meta)
    if jax.process_index() == 0:
        pointer = f"{run_dir}/ckpt/LATEST"
        previous = json.loads(read_bytes(pointer)) if os.path.exists(pointer) else {}
        write_bytes(pointer, json.dumps({"current": name, "previous": previous.get("current")}).encode())
        if previous.get("previous"):
            shutil.rmtree(f"{run_dir}/ckpt/{previous['previous']}", ignore_errors=True)
    return directory


def save_milestone(run_dir: str, step: int, trees: dict, meta: dict) -> str:
    directory = f"{run_dir}/milestones/step-{step:09d}"
    save(directory, trees, meta)
    return directory


def parse_steps(spec) -> set[int]:
    """``"3906, 39060"`` or ``[3906, 39060]`` -> ``{3906, 39060}``."""
    if isinstance(spec, int):
        return {spec}
    if isinstance(spec, str):
        spec = spec.replace(",", " ").split()
    return {int(x) for x in spec}

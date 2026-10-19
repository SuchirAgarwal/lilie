"""Shared training helpers for distributed setup, data-parallel sharding, schedules, EMA."""
from __future__ import annotations

import math
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# ------------------------------------------------------------------ distributed
def init_distributed(enabled: bool) -> None:
    """
    Initialise JAX's multi-host runtime on TPU pods (one process per host).
    """
    if not enabled:
        return
    env = os.environ.get
    jax.distributed.initialize(
        coordinator_address=env("COORDINATOR_ADDRESS"),
        num_processes=int(env("NUM_PROCESSES")) if env("NUM_PROCESSES") else None,
        process_id=int(env("PROCESS_ID")) if env("PROCESS_ID") else None,
    )


def is_primary() -> bool:
    return jax.process_index() == 0


def log(message: str) -> None:
    """Print from process 0 only."""
    if is_primary():
        print(message, flush=True)


def set_seed(seed: int):
    """Seed the host RNGs (offset per process, since each host owns its own data stream)
    and return the process-invariant root JAX key."""
    random.seed(seed + jax.process_index())
    np.random.seed(seed + jax.process_index())
    return jax.random.key(seed)


# --------------------------------------------------------------------- sharding
def make_mesh() -> Mesh:
    """One-dimensional data-parallel mesh over every device of every host."""
    return Mesh(np.asarray(jax.devices(), dtype=object), ("data",))


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec("data"))


def replicate(mesh: Mesh, tree):
    """Place a pytree (parameters, optimizer state) replicated on every device."""
    sharding = NamedSharding(mesh, PartitionSpec())

    def place(x):
        if sharding.is_fully_addressable:
            return jax.device_put(x, sharding)
        return jax.make_array_from_process_local_data(sharding, np.asarray(x), tuple(x.shape))

    return jax.tree.map(place, tree)


def shard_batch(mesh: Mesh, tree):
    """Each process contributes its local batch; the result is one globally sharded batch
    whose leading dimension is ``local_batch * process_count``."""
    sharding = batch_sharding(mesh)

    def place(x):
        if sharding.is_fully_addressable:
            return jax.device_put(x, sharding)
        return jax.make_array_from_process_local_data(sharding, np.asarray(x))

    return jax.tree.map(place, tree)


def shard_global_batch(mesh: Mesh, tree):
    """Shard a batch that every process holds identically (e.g. fixed sample captions)."""
    sharding = batch_sharding(mesh)

    def place(x):
        x = np.asarray(x)
        if sharding.is_fully_addressable:
            return jax.device_put(x, sharding)
        return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])

    return jax.tree.map(place, tree)


def gather(value) -> np.ndarray:
    """The full (global) value of a sharded array on every host."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value)


def local_replica(value) -> np.ndarray:
    """One host copy of a replicated array, without a cross-host gather."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        value = value.addressable_data(0)
    return np.asarray(value)


# -------------------------------------------------------------------- schedules
def cosine_lr(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float) -> float:
    """Linear warmup from ``min_lr`` to ``max_lr``, then cosine decay back to ``min_lr``."""
    if step <= 0:
        return min_lr
    if step <= warmup_steps:
        return min_lr + step / warmup_steps * (max_lr - min_lr)
    t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


def warmup_flat_cooldown(step: int, warmup_steps: int, cooldown_steps: int, total_steps: int) -> float:
    """PixelGen's LR multiplier: linear warmup, constant, optional linear cooldown."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if step < total_steps - cooldown_steps:
        return 1.0
    return max(0.0, 1.0 - (step - (total_steps - cooldown_steps)) / max(1, cooldown_steps))


class AutoClip:
    """Adaptive gradient clipping (Seetharaman et al., 2020): clip each step to the p-th
    percentile of every pre-clip gradient norm observed so far (one slot per step)."""

    def __init__(self, percentile: float, total_steps: int):
        self.percentile = float(percentile)
        self.history = np.zeros((total_steps,), np.float32)
        self.count = 0

    def observe(self, grad_norm: float) -> float:
        """Record this step's norm and return the clip threshold for this step."""
        if self.count == len(self.history):
            self.history = np.concatenate([self.history, np.zeros_like(self.history)])
        self.history[self.count] = grad_norm
        self.count += 1
        return float(np.quantile(self.history[: self.count], self.percentile / 100))

    def state(self) -> dict:
        return {"history": self.history, "count": np.asarray(self.count, np.int32)}

    def load(self, state: dict) -> None:
        """Restore a saved history into this buffer (which may be sized for a new ``total_steps``)."""
        self.count = int(state["count"])
        if self.count > len(self.history):
            self.history = np.zeros((self.count,), np.float32)
        self.history[: self.count] = np.asarray(state["history"], np.float32)[: self.count]


def synchronized_batches(iterator, n: int):
    """Yield up to ``n`` batches, stopping on EVERY process as soon as any process runs out,
    so a short validation stream on one host cannot leave the others waiting in a collective."""
    for _ in range(n):
        batch = next(iterator, None)
        if not all_processes(batch is not None):
            return
        yield batch


def all_processes(flag: bool) -> bool:
    """True when ``flag`` is true on every process."""
    if jax.process_count() == 1:
        return bool(flag)
    from jax.experimental import multihost_utils
    return bool(np.min(multihost_utils.process_allgather(np.asarray([flag], np.int32), tiled=True)))


# -------------------------------------------------------------------------- EMA
def init_ema(params):
    return jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32, copy=True), params)


def update_ema(ema, params, decay: float):
    d, one_minus = jnp.float32(decay), jnp.float32(1.0 - decay)
    return jax.tree.map(lambda e, p: d * e + one_minus * p.astype(jnp.float32), ema, params)



def count_params(tree) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tree))

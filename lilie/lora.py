"""
LoRA for SmolVLM
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class LoRAParam(nnx.Param):
    """Marker type for adapter tensors."""


class LoRALinear(nnx.Module):
    def __init__(self, base: nnx.Module, rank: int, alpha: float, *, rngs: nnx.Rngs):
        in_features, out_features = (int(d) for d in base.kernel.value.shape)
        self.base = base  # frozen
        self.scale = alpha / rank
        bound = 1.0 / math.sqrt(in_features)
        self.lora_A = LoRAParam(jax.random.uniform(
            rngs.params(), (rank, in_features), jnp.float32, minval=-bound, maxval=bound))
        self.lora_B = LoRAParam(jnp.zeros((out_features, rank), jnp.float32))

    def __call__(self, x):
        a = self.lora_A.value.astype(x.dtype)
        b = self.lora_B.value.astype(x.dtype)
        return self.base(x) + (x @ a.T @ b.T) * self.scale


def _is_linear(module) -> bool:
    kernel = getattr(module, "kernel", None)
    return (isinstance(module, nnx.Module) and not isinstance(module, LoRALinear)
            and isinstance(kernel, nnx.Param) and kernel.value.ndim == 2)


def inject_lora(model: nnx.Module, rank: int, alpha: float, *, rngs: nnx.Rngs) -> int:
    """Wrap every leaf linear layer except ``model.lm_head`` in place; returns the count."""
    head = getattr(model, "lm_head", None)
    count = 0
    visited = set()

    def visit(node):
        nonlocal count
        if id(node) in visited:
            return
        visited.add(id(node))
        children = (vars(node).items() if isinstance(node, nnx.Module)
                    else enumerate(node) if isinstance(node, (list, tuple)) else [])
        for key, child in list(children):
            if isinstance(key, str) and key.startswith("_"):
                continue
            if isinstance(child, LoRALinear):
                continue
            if _is_linear(child) and child is not head:
                wrapped = LoRALinear(child, rank, alpha, rngs=rngs)
                if isinstance(node, nnx.Module):
                    setattr(node, key, wrapped)
                else:
                    node[key] = wrapped
                count += 1
            elif isinstance(child, (nnx.Module, list, tuple)):
                visit(child)

    visit(model)
    assert count, "no linear layers found"
    return count

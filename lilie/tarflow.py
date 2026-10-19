"""
The LILIE encoder: improved TARFlow (Zhai et al., 2024; BiFlow, Lu et al., 2025).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def _layer_norm_stats(x):
    xf = x.astype(jnp.float32)
    mu = xf.mean(axis=-1, keepdims=True)
    var = jnp.maximum(0.0, (xf * xf).mean(axis=-1, keepdims=True) - mu * mu)
    return x - mu, jax.lax.rsqrt(var + 1e-6)


def _layer_norm_no_params(x, dtype):
    y, mul = _layer_norm_stats(x)
    return (y * mul).astype(dtype)


class PermutationFlip(nnx.Module):
    """Self-inverse reversal of the token axis (applied on the way in and out of a block)."""

    def __init__(self, flip: bool):
        self.flip = flip

    def __call__(self, x):
        return jnp.flip(x, axis=1) if self.flip else x


class Linear(nnx.Module):
    """Dense layer, Xavier-uniform kernel ``(in, out)`` and zero bias, fp32 parameters."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.kernel = nnx.Param(nnx.initializers.xavier_uniform()(
            rngs.params(), (in_features, out_features), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((out_features,), jnp.float32))

    def __call__(self, x, dtype):
        return jnp.matmul(x.astype(dtype), self.kernel[...].astype(dtype)) + self.bias[...].astype(dtype)


class LayerNorm(nnx.Module):
    def __init__(self, num_features: int):
        self.scale = nnx.Param(jnp.ones((num_features,), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_features,), jnp.float32))

    def __call__(self, x, dtype):
        y, mul = _layer_norm_stats(x)
        return (y * (mul * self.scale[...]) + self.bias[...]).astype(dtype)


def _attention(q, k, v, mask, dtype):
    """Masked softmax attention in BTHD layout; ``mask`` selects the keys each query may see."""
    q = q / jnp.sqrt(q.shape[-1]).astype(dtype)
    scores = jnp.einsum("...qhd,...khd->...hqk", q, k)
    scores = jnp.where(mask, scores, jnp.finfo(dtype).min)
    weights = jax.nn.softmax(scores, axis=-1).astype(dtype)
    return jnp.einsum("...hqk,...khd->...qhd", weights, v)


class Attention(nnx.Module):
    """Pre-norm causal self-attention with ``channels // 64`` heads."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.num_heads = max(1, channels // 64)
        self.head_dim = channels // self.num_heads
        assert self.head_dim * self.num_heads == channels
        self.qkv = Linear(channels, 3 * channels, rngs=rngs)
        self.proj = Linear(channels, channels, rngs=rngs)
        self.norm = LayerNorm(channels)

    def _qkv_heads(self, x, dtype):
        B, T, C = x.shape
        h = self.norm(x.astype(dtype), dtype)
        q, k, v = jnp.split(self.qkv(h, dtype), 3, axis=-1)
        shape = (B, T, self.num_heads, self.head_dim)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def __call__(self, x, mask, *, dtype):
        B, T, C = x.shape
        q, k, v = self._qkv_heads(x, dtype)
        return self.proj(_attention(q, k, v, mask, dtype).reshape(B, T, C), dtype)

    def cached_step(self, x, k_cache, v_cache, index, *, dtype):
        """One token at position ``index``, attending to the cached prefix (for the inverse)."""
        B, S, C = x.shape
        q, k, v = self._qkv_heads(x, dtype)
        k_cache = jax.lax.dynamic_update_slice(k_cache, k.astype(k_cache.dtype), (0, index, 0, 0))
        v_cache = jax.lax.dynamic_update_slice(v_cache, v.astype(v_cache.dtype), (0, index, 0, 0))
        valid = jnp.arange(k_cache.shape[1]) <= index
        return self.proj(_attention(q, k_cache, v_cache, valid, dtype).reshape(B, S, C), dtype), k_cache, v_cache


class MLP(nnx.Module):
    """Pre-norm MLP, expansion 4, exact GELU."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.norm = LayerNorm(channels)
        self.net1 = Linear(channels, 4 * channels, rngs=rngs)
        self.net2 = Linear(4 * channels, channels, rngs=rngs)

    def __call__(self, x, *, dtype):
        h = self.norm(x.astype(dtype), dtype)
        return self.net2(jax.nn.gelu(self.net1(h, dtype), approximate=False), dtype)


class Layer(nnx.Module):
    """One Transformer layer. The residual stream stays fp32; a parameter-free LayerNorm
    precedes the MLP's own LayerNorm (kept as in the reference implementation)."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.attn = Attention(channels, rngs=rngs)
        self.mlp = MLP(channels, rngs=rngs)

    def __call__(self, x, mask, *, dtype):
        x = x + self.attn(x, mask, dtype=dtype)
        return x + self.mlp(_layer_norm_no_params(x, dtype), dtype=dtype)

    def cached_step(self, x, k_cache, v_cache, index, *, dtype):
        attn, k_cache, v_cache = self.attn.cached_step(x, k_cache, v_cache, index, dtype=dtype)
        x = x + attn
        return x + self.mlp(_layer_norm_no_params(x, dtype), dtype=dtype), k_cache, v_cache


class MetaBlock(nnx.Module):
    """One autoregressive affine coupling block."""

    def __init__(self, in_dim: int, channels: int, num_patches: int, num_layers: int,
                 permutation: PermutationFlip, clip_range: float, *, rngs: nnx.Rngs):
        self.num_patches = num_patches
        self.clip_range = clip_range
        self.proj_in = Linear(in_dim, channels, rngs=rngs)
        self.pos_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, num_patches, channels), jnp.float32))
        self.layers = nnx.List([Layer(channels, rngs=rngs) for _ in range(num_layers)])
        # Zero-initialised head: every block is exactly the identity at initialisation.
        self.proj_out = Linear(channels, 2 * in_dim, rngs=rngs)
        self.proj_out.kernel[...] = jnp.zeros_like(self.proj_out.kernel[...])
        self.permutation = permutation

    def _head(self, h, dtype):
        return self.proj_out(_layer_norm_no_params(h, dtype), dtype)

    def __call__(self, x, *, dtype):
        """Forward coupling on ``(B, T, D)`` tokens; returns ``(x_new, log_jacobian)``."""
        x_in = self.permutation(x)
        h = self.proj_in(x_in, dtype) + self.permutation(self.pos_embedding[...])
        mask = jnp.tril(jnp.ones((self.num_patches, self.num_patches), dtype=bool))
        for layer in self.layers:
            h = layer(h, mask, dtype=dtype)
        h = self._head(h, dtype)
        h = jnp.concatenate([jnp.zeros_like(h[:, :1]), h[:, :-1]], axis=1)  # token t sees x_{<t}
        h = jnp.clip(h, -self.clip_range, self.clip_range)
        alpha, mu = jnp.split(h, 2, axis=-1)
        x_new = (x_in - mu) * jnp.exp(-alpha)
        return self.permutation(x_new), -alpha.mean(axis=(1, 2))

    def inverse(self, z, *, dtype=jnp.float32):
        """Exact inverse, one token per step with a KV cache (``lax.scan`` over positions)."""
        x = self.permutation(z.astype(jnp.float32))
        pos = self.permutation(self.pos_embedding[...])
        B, T, _ = x.shape
        assert T == self.num_patches
        keys = tuple(jnp.zeros((B, T, l.attn.num_heads, l.attn.head_dim), dtype) for l in self.layers)
        values = tuple(jnp.zeros_like(k) for k in keys)

        def body(carry, i):
            x, keys, values = carry
            h = self.proj_in(jax.lax.dynamic_slice_in_dim(x, i, 1, axis=1), dtype)
            h = h + jax.lax.dynamic_slice_in_dim(pos, i, 1, axis=1)
            new_keys, new_values = [], []
            for li, layer in enumerate(self.layers):
                h, k, v = layer.cached_step(h, keys[li], values[li], i, dtype=dtype)
                new_keys.append(k)
                new_values.append(v)
            out = jnp.clip(self._head(h, dtype), -self.clip_range, self.clip_range)
            alpha, mu = jnp.split(out, 2, axis=-1)
            x_next = jax.lax.dynamic_slice_in_dim(x, i + 1, 1, axis=1)
            x_next = x_next * jnp.exp(alpha.astype(jnp.float32)) + mu
            x = jax.lax.dynamic_update_slice_in_dim(x, x_next, i + 1, axis=1)
            return (x, tuple(new_keys), tuple(new_values)), None

        (x, _, _), _ = jax.lax.scan(body, (x, keys, values), jnp.arange(T - 1, dtype=jnp.int32))
        return self.permutation(x)


class Model(nnx.Module):
    """The full encoder R_theta: NCHW image in [-1, 1] -> ``(B, T, D)`` patch tokens.

    ``T = (img_size / patch_size)**2`` tokens of ``D = 3 * patch_size**2`` values. The output
    is the same size as the input, and ``unpatchify`` turns it back into an NCHW image.
    """

    def __init__(self, img_size: int, patch_size: int, channels: int, num_blocks: int,
                 layers_per_block: int, clip_range: float = 1.0, *, rngs: nnx.Rngs,
                 compute_dtype=jnp.float32, remat: bool = True):
        assert img_size % patch_size == 0
        self.img_size, self.patch_size = img_size, patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.compute_dtype, self.remat = compute_dtype, remat
        in_dim = 3 * patch_size ** 2
        self.blocks = nnx.List([
            MetaBlock(in_dim, channels, self.num_patches, layers_per_block,
                      PermutationFlip(i % 2 == 1), clip_range, rngs=rngs)
            for i in range(num_blocks)])
        # Running global second moment E[z^2] of the encoder output, broadcast to (T, D). Saved
        # with the encoder weights; the generator's loss divides by sqrt of it.
        self.m2 = nnx.Variable(jnp.ones((self.num_patches, in_dim), jnp.float32))

    @property
    def latent_m2(self) -> float:
        return float(self.m2[...].reshape(-1)[0])

    def patchify(self, x):
        """(B, 3, H, W) -> (B, T, p*p*3), NHWC patch order."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.transpose(0, 2, 3, 1).reshape(B, H // p, p, W // p, p, C)
        return x.transpose(0, 1, 3, 2, 4, 5).reshape(B, self.num_patches, p * p * C)

    def unpatchify(self, z):
        """Exact inverse of :meth:`patchify`."""
        B, T, D = z.shape
        p, hp = self.patch_size, self.img_size // self.patch_size
        u = z.reshape(B, hp, hp, p, p, D // (p * p)).transpose(0, 1, 3, 2, 4, 5)
        return u.reshape(B, self.img_size, self.img_size, -1).transpose(0, 3, 1, 2)

    def _block_forward(self, block, x, dtype):
        if not self.remat:
            return block(x, dtype=dtype)
        graphdef, state = nnx.split(block)

        def fn(state, x):
            return nnx.merge(graphdef, state)(x, dtype=dtype)

        return jax.checkpoint(fn)(state, x)

    def __call__(self, x, *, compute_dtype=None):
        """Encode to fp32 tokens ``(B, T, D)``."""
        dtype = self.compute_dtype if compute_dtype is None else compute_dtype
        x = self.patchify(x)
        logdet = jnp.zeros((), jnp.float32)
        for block in self.blocks:
            x, block_logdet = self._block_forward(block, x, dtype)
            logdet = logdet + block_logdet
        return x.astype(jnp.float32), logdet

    def inverse(self, z, *, dtype=jnp.float32):
        """Decode fp32 tokens ``(B, T, D)`` back to an NCHW image (exact up to fp rounding)."""
        x = z.astype(jnp.float32)
        for block in reversed(self.blocks):
            x = block.inverse(x, dtype=dtype)
        return self.unpatchify(x)

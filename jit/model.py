"""
JiT-T2I (PixelGen without REPA or perceptual losses): a pixel-space x0-prediction Transformer conditioned on Qwen3 text embeddings.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# -------------------------------------------------------------- position tables
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.einsum("m,d->md", np.asarray(pos).reshape(-1), omega)
    return np.concatenate((np.sin(out), np.cos(out)), axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Row-major 2-D sin/cos table ``(grid_size**2, embed_dim)``."""
    grid = np.stack(np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)))
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate((emb_h, emb_w), axis=1)


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0) -> np.ndarray:
    """Interleaved axial RoPE table as complex64, ``(height*width, dim/2)``."""
    assert dim % 4 == 0, "the axial RoPE head dimension must be divisible by 4"
    y_axis = np.linspace(0.0, width, height, dtype=np.float32)
    x_axis = np.linspace(0.0, height, width, dtype=np.float32)
    y_pos, x_pos = np.meshgrid(y_axis, x_axis, indexing="ij")
    exponent = np.arange(0, dim, 4, dtype=np.float32)[: dim // 4] / np.float32(dim)
    freqs = np.float32(1.0) / np.power(np.float32(theta), exponent)
    x_angle = np.outer(x_pos.reshape(-1), freqs).astype(np.float32)
    y_angle = np.outer(y_pos.reshape(-1), freqs).astype(np.float32)
    x_cis = (np.cos(x_angle) + np.complex64(1j) * np.sin(x_angle)).astype(np.complex64)
    y_cis = (np.cos(y_angle) + np.complex64(1j) * np.sin(y_angle)).astype(np.complex64)
    return np.stack((x_cis, y_cis), axis=-1).reshape(height * width, dim // 2)


def _apply_rotary(x, freqs_cis):
    xf = x.astype(jnp.float32)
    pairs = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 2, 2)
    out = jax.lax.complex(pairs[..., 0], pairs[..., 1]) * freqs_cis[None, None]
    return jnp.stack((jnp.real(out), jnp.imag(out)), axis=-1).reshape(x.shape).astype(x.dtype)


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


# ------------------------------------------------------------------ primitives
class Linear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, rngs: nnx.Rngs,
                 kernel_init: str = "xavier"):
        key = rngs.params()
        shape = (in_features, out_features)
        if kernel_init == "xavier":
            kernel = jax.nn.initializers.xavier_uniform()(key, shape, jnp.float32)
        elif kernel_init == "normal_002":
            kernel = 0.02 * jax.random.normal(key, shape, jnp.float32)
        elif kernel_init == "zeros":
            kernel = jnp.zeros(shape, jnp.float32)
        else:
            raise ValueError(f"unknown kernel_init {kernel_init!r}")
        self.kernel = nnx.Param(kernel)
        if bias:
            self.bias = nnx.Param(jnp.zeros((out_features,), jnp.float32))
        else:
            self.bias = None

    def __call__(self, x, compute_dtype):
        y = x.astype(compute_dtype) @ self.kernel[...].astype(compute_dtype)
        return y if self.bias is None else y + self.bias[...].astype(compute_dtype)


class RMSNorm(nnx.Module):
    """fp32 statistics and affine, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(jnp.float32)
        h = h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + self.eps)
        return (self.weight[...] * h).astype(x.dtype)


class SwiGLU(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.w12 = Linear(dim, hidden_dim * 2, bias=False, rngs=rngs)
        self.w3 = Linear(hidden_dim, dim, bias=False, rngs=rngs)

    def __call__(self, x, compute_dtype):
        x1, x2 = jnp.split(self.w12(x, compute_dtype), 2, axis=-1)
        return self.w3(jax.nn.silu(x1) * x2, compute_dtype)


class Sequential(nnx.Module):
    """Parameter container with the torch ``nn.Sequential`` names ``"0"``, ``"2"``."""

    def __init__(self, **layers):
        for name, layer in layers.items():
            setattr(self, name, layer)


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs):
        self.mlp = Sequential(**{
            "0": Linear(frequency_embedding_size, hidden_size, rngs=rngs, kernel_init="normal_002"),
            "2": Linear(hidden_size, hidden_size, rngs=rngs, kernel_init="normal_002")})
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t, compute_dtype):
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(-math.log(10.0) * jnp.arange(half, dtype=jnp.float32) / half)  # max_period 10
        args = t[..., None].astype(jnp.float32) * freqs
        h = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
        h = jax.nn.silu(getattr(self.mlp, "0")(h, compute_dtype))
        return getattr(self.mlp, "2")(h, compute_dtype)


class Embed(nnx.Module):
    """Text projection followed by RMSNorm."""

    def __init__(self, in_dim: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.proj = Linear(in_dim, embed_dim, bias=True, rngs=rngs)
        self.norm = RMSNorm(embed_dim)

    def __call__(self, x, compute_dtype):
        return self.norm(self.proj(x, compute_dtype))


def _sdpa(q, k, v):
    """``(B, H, S, D)`` attention with fp32 scores and softmax."""
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * q.shape[-1] ** -0.5
    probs = jax.nn.softmax(scores, axis=-1).astype(q.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v)


class Attention(nnx.Module):
    """Image self-attention (with 2-D RoPE) fused with cross-attention to all text tokens."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv_x = Linear(dim, dim * 3, bias=False, rngs=rngs)
        self.kv_y = Linear(dim, dim * 2, bias=False, rngs=rngs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = Linear(dim, dim, bias=True, rngs=rngs)

    def __call__(self, x, y, pos, compute_dtype):
        B, N, C = x.shape
        qkv = self.qkv_x(x, compute_dtype).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, kx = _apply_rotary(self.q_norm(qkv[0]), pos), _apply_rotary(self.k_norm(qkv[1]), pos)
        kv = self.kv_y(y, compute_dtype).reshape(B, y.shape[1], 2, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        k = jnp.concatenate((kx, self.k_norm(kv[0])), axis=2)
        v = jnp.concatenate((qkv[2], kv[1]), axis=2)
        return self.proj(_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(B, N, C), compute_dtype)


class TextRefineAttention(nnx.Module):
    """Unmasked text self-attention."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = Linear(dim, dim * 3, bias=False, rngs=rngs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = Linear(dim, dim, bias=True, rngs=rngs)

    def __call__(self, x, compute_dtype):
        B, N, C = x.shape
        qkv = self.qkv(x, compute_dtype).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        out = _sdpa(self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2])
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, C), compute_dtype)


class FlattenDiTBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio), rngs=rngs)
        self.adaLN_modulation = Sequential(**{"0": Linear(hidden_size, 6 * hidden_size, rngs=rngs, kernel_init="zeros")})

    def __call__(self, x, y, c, pos, compute_dtype):
        mod = getattr(self.adaLN_modulation, "0")(c, compute_dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), y, pos, compute_dtype)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), compute_dtype)


class TextRefineBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(hidden_size)
        self.attn = TextRefineAttention(hidden_size, num_heads, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio), rngs=rngs)
        self.adaLN_modulation = Sequential(**{"0": Linear(hidden_size, 6 * hidden_size, rngs=rngs, kernel_init="zeros")})

    def __call__(self, x, c, compute_dtype):
        mod = getattr(self.adaLN_modulation, "0")(c, compute_dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), compute_dtype)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), compute_dtype)


class BottleneckEmbed(nnx.Module):
    def __init__(self, patch_dim: int, embed_dim: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.proj1 = Linear(patch_dim, bottleneck_dim, bias=False, rngs=rngs)
        self.proj2 = Linear(bottleneck_dim, embed_dim, bias=True, rngs=rngs)

    def __call__(self, x, compute_dtype):
        return self.proj2(self.proj1(x, compute_dtype), compute_dtype)


class FinalLayer(nnx.Module):
    def __init__(self, hidden_size: int, out_dim: int, *, rngs: nnx.Rngs):
        self.linear = Linear(hidden_size, out_dim, bias=True, rngs=rngs, kernel_init="zeros")

    def __call__(self, x, compute_dtype):
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        return self.linear(((xf - mean) * jax.lax.rsqrt(var + 1e-6)).astype(x.dtype), compute_dtype)


# ------------------------------------------------------------------- the model
class JiT_T2I(nnx.Module):
    def __init__(self, *, input_size: int = 256, patch_size: int = 16, hidden_size: int = 1536,
                 num_heads: int = 24, num_blocks: int = 16, num_text_blocks: int = 4,
                 bottleneck_dim: int = 256, mlp_ratio: float = 4.0, txt_embed_dim: int = 2048,
                 txt_max_length: int = 300, rngs: nnx.Rngs, compute_dtype=jnp.bfloat16, remat: bool = True):
        assert input_size % patch_size == 0 and hidden_size % num_heads == 0 and (hidden_size // num_heads) % 4 == 0
        self.input_size, self.patch_size, self.hidden_size = input_size, patch_size, hidden_size
        self.txt_max_length, self.txt_embed_dim = txt_max_length, txt_embed_dim
        self.compute_dtype, self.remat = compute_dtype, remat
        patch_dim = 3 * patch_size ** 2
        grid = input_size // patch_size
        self.x_embedder = BottleneckEmbed(patch_dim, hidden_size, bottleneck_dim, rngs=rngs)
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, rngs=rngs)
        self.pos_embed = nnx.Variable(jnp.asarray(get_2d_sincos_pos_embed(hidden_size, grid)[None], jnp.float32))
        self.y_pos_embedding = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, txt_max_length, hidden_size), jnp.float32))
        self.xpos = nnx.Variable(jnp.asarray(precompute_freqs_cis_2d(hidden_size // num_heads, grid, grid), jnp.complex64))
        self.blocks = nnx.List([FlattenDiTBlock(hidden_size, num_heads, mlp_ratio, rngs=rngs) for _ in range(num_blocks)])
        self.text_refine_blocks = nnx.List([TextRefineBlock(hidden_size, num_heads, mlp_ratio, rngs=rngs) for _ in range(num_text_blocks)])
        self.final_layer = FinalLayer(hidden_size, patch_dim, rngs=rngs)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        u = x.reshape(B, C, H // p, p, W // p, p).transpose(0, 2, 4, 1, 3, 5)
        return u.reshape(B, (H // p) * (W // p), C * p * p)

    def unpatchify(self, x):
        B, T, D = x.shape
        p, g = self.patch_size, self.input_size // self.patch_size
        u = x.reshape(B, g, g, 3, p, p).transpose(0, 3, 1, 4, 2, 5)
        return u.reshape(B, 3, self.input_size, self.input_size)

    def _run(self, block, args, compute_dtype):
        if not self.remat:
            return block(*args, compute_dtype)
        graphdef, state = nnx.split(block)
        return jax.checkpoint(lambda s, *a: nnx.merge(graphdef, s)(*a, compute_dtype))(state, *args)

    def __call__(self, x, t, y, *, compute_dtype=None):
        """``x`` noisy image (B, 3, H, W); ``t`` (B,) in [0, 1]; ``y`` text states
        (B, txt_max_length, txt_embed_dim). Returns the predicted clean image."""
        dtype = self.compute_dtype if compute_dtype is None else compute_dtype
        B = x.shape[0]
        t = self.t_embedder(t.reshape(-1), dtype).reshape(B, 1, self.hidden_size)
        c = jax.nn.silu(t)
        y = self.y_embedder(y, dtype) + self.y_pos_embedding[...].astype(dtype)
        for block in self.text_refine_blocks:
            y = self._run(block, (y, c), dtype)
        h = (self.x_embedder(self.patchify(x), dtype) + self.pos_embed[...]).astype(dtype)
        for block in self.blocks:
            h = self._run(block, (h, y, c, self.xpos[...]), dtype)
        return self.unpatchify(self.final_layer(h, dtype))

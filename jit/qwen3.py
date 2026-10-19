"""
Frozen Qwen3-1.7B text encoder
"""
from __future__ import annotations

import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from common import hf


@dataclasses.dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 151_936


def config_from_hf(raw: dict) -> Qwen3Config:
    rope = raw.get("rope_parameters") or {}
    return Qwen3Config(
        hidden_size=raw["hidden_size"], num_hidden_layers=raw["num_hidden_layers"],
        intermediate_size=raw["intermediate_size"], num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
        rms_norm_eps=raw["rms_norm_eps"], rope_theta=rope.get("rope_theta", raw.get("rope_theta")),
        vocab_size=raw["vocab_size"])


class Linear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, dtype):
        self.kernel = nnx.Param(jnp.zeros((in_features, out_features), dtype))

    def __call__(self, x, compute_dtype):
        return x.astype(compute_dtype) @ self.kernel[...].astype(compute_dtype)


class Embedding(nnx.Module):
    def __init__(self, num_embeddings: int, dim: int, dtype):
        self.weight = nnx.Param(jnp.zeros((num_embeddings, dim), dtype))

    def __call__(self, ids, compute_dtype):
        return jnp.take(self.weight[...], ids, axis=0).astype(compute_dtype)


class Qwen3RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(jnp.float32)
        h = h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + self.eps)
        return h.astype(x.dtype) * self.weight[...].astype(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class Qwen3RotaryEmbedding(nnx.Module):
    def __init__(self, cfg: Qwen3Config):
        exponent = np.arange(0, cfg.head_dim, 2, dtype=np.float32) / np.float32(cfg.head_dim)
        self.inv_freq = nnx.Variable(jnp.asarray(1.0 / np.power(np.float32(cfg.rope_theta), exponent), jnp.float32))

    def __call__(self, position_ids, compute_dtype):
        freqs = position_ids.astype(jnp.float32)[..., None] * self.inv_freq[...][None, None, :]
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb).astype(compute_dtype), jnp.sin(emb).astype(compute_dtype)


def _attention(q, k, v, attention_mask, groups):
    """Causal, right-padded grouped-query attention on ``(B, S, H, D)`` inputs."""
    seq = q.shape[1]
    k = jnp.repeat(k, groups, axis=2).transpose(0, 2, 1, 3)
    v = jnp.repeat(v, groups, axis=2).transpose(0, 2, 1, 3)
    q = q.transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * q.shape[-1] ** -0.5
    mask = (jnp.arange(seq)[None, :] <= jnp.arange(seq)[:, None])[None, None] & attention_mask[:, None, None, :]
    scores = jnp.where(mask, scores, -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(scores, axis=-1), nan=0.0).astype(q.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v).transpose(0, 2, 1, 3)


class Qwen3Attention(nnx.Module):
    def __init__(self, cfg: Qwen3Config, dtype):
        d, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = Linear(d, cfg.num_attention_heads * hd, dtype)
        self.k_proj = Linear(d, cfg.num_key_value_heads * hd, dtype)
        self.v_proj = Linear(d, cfg.num_key_value_heads * hd, dtype)
        self.o_proj = Linear(cfg.num_attention_heads * hd, d, dtype)
        self.q_norm = Qwen3RMSNorm(hd, cfg.rms_norm_eps, dtype)
        self.k_norm = Qwen3RMSNorm(hd, cfg.rms_norm_eps, dtype)
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, hd

    def __call__(self, x, cos, sin, attention_mask, compute_dtype):
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x, compute_dtype).reshape(B, S, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x, compute_dtype).reshape(B, S, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, compute_dtype).reshape(B, S, self.num_kv_heads, self.head_dim)
        c, s = cos[:, :, None, :], sin[:, :, None, :]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        out = _attention(q, k, v, attention_mask, self.num_heads // self.num_kv_heads)
        return self.o_proj(out.reshape(B, S, -1), compute_dtype)


class Qwen3MLP(nnx.Module):
    def __init__(self, cfg: Qwen3Config, dtype):
        self.gate_proj = Linear(cfg.hidden_size, cfg.intermediate_size, dtype)
        self.up_proj = Linear(cfg.hidden_size, cfg.intermediate_size, dtype)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, dtype)

    def __call__(self, x, compute_dtype):
        return self.down_proj(jax.nn.silu(self.gate_proj(x, compute_dtype)) * self.up_proj(x, compute_dtype), compute_dtype)


class Qwen3DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3Config, dtype):
        self.self_attn = Qwen3Attention(cfg, dtype)
        self.mlp = Qwen3MLP(cfg, dtype)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)

    def __call__(self, x, cos, sin, attention_mask, compute_dtype):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask, compute_dtype)
        return x + self.mlp(self.post_attention_layernorm(x), compute_dtype)


class Qwen3Model(nnx.Module):
    def __init__(self, cfg: Qwen3Config, dtype):
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, dtype)
        self.layers = nnx.List([Qwen3DecoderLayer(cfg, dtype) for _ in range(cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.rotary_emb = Qwen3RotaryEmbedding(cfg)

    def __call__(self, input_ids, attention_mask, compute_dtype):
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids, compute_dtype)
        cos, sin = self.rotary_emb(jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32)[None], (B, S)), compute_dtype)
        mask = attention_mask.astype(bool)
        for layer in self.layers:
            x = layer(x, cos, sin, mask, compute_dtype)
        return self.norm(x)


class Qwen3(nnx.Module):

    def __init__(self, cfg: Qwen3Config, dtype=jnp.bfloat16):
        self.model = Qwen3Model(cfg, dtype)
        self.cfg = cfg

    def __call__(self, input_ids, attention_mask, *, compute_dtype=jnp.bfloat16):
        return self.model(input_ids, attention_mask, compute_dtype)


def build_qwen3(model_name: str = "Qwen/Qwen3-1.7B") -> tuple[Qwen3, Qwen3Config]:
    snapshot = hf.snapshot(model_name)
    with open(os.path.join(snapshot, "config.json")) as handle:
        cfg = config_from_hf(json.load(handle))
    tensors = hf.load_snapshot_tensors(snapshot)
    tensors.pop("lm_head.weight", None)
    model = Qwen3(cfg, jnp.bfloat16)
    hf.load_hf_state(model, tensors, jnp.bfloat16)
    return model, cfg


def load_tokenizer(model_name: str = "Qwen/Qwen3-1.7B"):
    from transformers import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(hf.snapshot(model_name), local_files_only=True)
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize(tokenizer, captions: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-width int32 ``(ids, attention_mask)``, truncated / right-padded to ``max_length``."""
    encoded = tokenizer(list(captions), truncation=True, max_length=max_length, padding="max_length",
                        return_attention_mask=True, return_tensors="np")
    return (np.asarray(encoded["input_ids"], dtype=np.int32),
            np.asarray(encoded["attention_mask"], dtype=np.int32))

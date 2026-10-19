"""
SmolVLM-256M-Instruct (Hugging Face ``Idefics3ForConditionalGeneration``) in Flax NNX.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# ------------------------------------------------------------------ configuration
@dataclasses.dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_channels: int = 3
    image_size: int = 512
    patch_size: int = 16
    layer_norm_eps: float = 1e-6


@dataclasses.dataclass(frozen=True)
class TextConfig:
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49280
    rope_theta: float = 100000.0


@dataclasses.dataclass(frozen=True)
class SmolVLMConfig:
    vision: VisionConfig
    text: TextConfig
    scale_factor: int = 4
    image_token_id: int = 49190


def config_from_hf(d: dict) -> SmolVLMConfig:
    """Build the config from the checkpoint's ``config.json``."""
    v, t = d["vision_config"], d["text_config"]
    rope_theta = t.get("rope_parameters", {}).get("rope_theta", t.get("rope_theta"))
    return SmolVLMConfig(
        vision=VisionConfig(
            hidden_size=v["hidden_size"], num_hidden_layers=v["num_hidden_layers"],
            num_attention_heads=v["num_attention_heads"], intermediate_size=v["intermediate_size"],
            num_channels=v["num_channels"], image_size=v["image_size"],
            patch_size=v["patch_size"], layer_norm_eps=v["layer_norm_eps"]),
        text=TextConfig(
            hidden_size=t["hidden_size"], num_hidden_layers=t["num_hidden_layers"],
            num_attention_heads=t["num_attention_heads"], num_key_value_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"], intermediate_size=t["intermediate_size"],
            rms_norm_eps=t["rms_norm_eps"], vocab_size=t["vocab_size"], rope_theta=rope_theta),
        scale_factor=d["scale_factor"], image_token_id=d["image_token_id"])


# ------------------------------------------------------------------- primitives
class Linear(nnx.Module):
    """``y = x @ kernel + bias`` with ``kernel`` = the transposed torch weight, ``(in, out)``."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool, dtype):
        self.kernel = nnx.Param(jnp.zeros((in_features, out_features), dtype))
        self.bias = nnx.Param(jnp.zeros((out_features,), dtype)) if use_bias else None
        self.dtype = dtype

    def __call__(self, x):
        x = x.astype(self.dtype)
        y = x @ self.kernel[...].astype(x.dtype)
        return y if self.bias is None else y + self.bias[...].astype(y.dtype)


class Embedding(nnx.Module):
    def __init__(self, num_embeddings: int, dim: int, dtype):
        self.weight = nnx.Param(jnp.zeros((num_embeddings, dim), dtype))

    def __call__(self, ids):
        return jnp.take(self.weight[...], ids, axis=0)


class LayerNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(jnp.float32)
        mu = jnp.mean(h, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(h - mu), axis=-1, keepdims=True)
        h = (h - mu) * jax.lax.rsqrt(var + self.eps)
        return h.astype(x.dtype) * self.weight[...].astype(x.dtype) + self.bias[...].astype(x.dtype)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype))
        self.eps = eps

    def __call__(self, x):
        h = x.astype(jnp.float32)
        h = h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + self.eps)
        return self.weight[...].astype(x.dtype) * h.astype(x.dtype)


def _sdpa(q, k, v, mask, scale):
    """Attention with fp32 scores and softmax; ``mask`` is boolean (True = attend) or None."""
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
    if mask is not None:
        s = jnp.where(mask, s, jnp.finfo(jnp.float32).min)
    p = jax.nn.softmax(s, axis=-1).astype(q.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v)


# ----------------------------------------------------------------- vision tower
class PatchEmbedding(nnx.Module):
    """``Conv2d(3, 768, kernel=stride=16)`` with the torch OIHW weight, computed as a matmul."""

    def __init__(self, vc: VisionConfig, dtype):
        self.weight = nnx.Param(jnp.zeros((vc.hidden_size, vc.num_channels, vc.patch_size, vc.patch_size), dtype))
        self.bias = nnx.Param(jnp.zeros((vc.hidden_size,), dtype))
        self.stride = vc.patch_size
        self.dtype = dtype

    def __call__(self, x):  # (B, 3, H, W) -> (B, 768, H/16, W/16)
        x = x.astype(self.dtype)
        B, C, H, W = x.shape
        P = self.stride
        w = self.weight[...].astype(x.dtype)
        xp = x.reshape(B, C, H // P, P, W // P, P).transpose(0, 2, 4, 1, 3, 5)
        y = jnp.einsum("bhwk,ok->bohw", xp.reshape(B, H // P, W // P, C * P * P), w.reshape(w.shape[0], -1))
        return y + self.bias[...].astype(y.dtype)[None, :, None, None]


class Idefics3VisionEmbeddings(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        self.patch_embedding = PatchEmbedding(vc, dtype)
        self.position_embedding = Embedding((vc.image_size // vc.patch_size) ** 2, vc.hidden_size, dtype)

    def __call__(self, pixel_values):
        x = self.patch_embedding(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)  # (B, S, 768)
        return x + self.position_embedding(jnp.arange(x.shape[1], dtype=jnp.int32)[None, :])


class Idefics3VisionAttention(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        d = vc.hidden_size
        self.q_proj = Linear(d, d, True, dtype)
        self.k_proj = Linear(d, d, True, dtype)
        self.v_proj = Linear(d, d, True, dtype)
        self.out_proj = Linear(d, d, True, dtype)
        self.num_heads = vc.num_attention_heads
        self.head_dim = d // vc.num_attention_heads

    def __call__(self, x):
        B, S, D = x.shape
        split = lambda t: t.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = _sdpa(split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)), None, self.head_dim ** -0.5)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, S, D))


class Idefics3VisionMLP(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        self.fc1 = Linear(vc.hidden_size, vc.intermediate_size, True, dtype)
        self.fc2 = Linear(vc.intermediate_size, vc.hidden_size, True, dtype)

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=True))


class Idefics3EncoderLayer(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        self.layer_norm1 = LayerNorm(vc.hidden_size, vc.layer_norm_eps, dtype)
        self.self_attn = Idefics3VisionAttention(vc, dtype)
        self.layer_norm2 = LayerNorm(vc.hidden_size, vc.layer_norm_eps, dtype)
        self.mlp = Idefics3VisionMLP(vc, dtype)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Idefics3Encoder(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        self.layers = nnx.List([Idefics3EncoderLayer(vc, dtype) for _ in range(vc.num_hidden_layers)])
        self.remat = False  # set before the first trace to recompute layers in backward

    def __call__(self, x):
        for layer in self.layers:
            if self.remat:
                graphdef, state = nnx.split(layer)
                x = jax.checkpoint(lambda s, h: nnx.merge(graphdef, s)(h))(state, x)
            else:
                x = layer(x)
        return x


class Idefics3VisionTransformer(nnx.Module):
    def __init__(self, vc: VisionConfig, dtype):
        self.embeddings = Idefics3VisionEmbeddings(vc, dtype)
        self.encoder = Idefics3Encoder(vc, dtype)
        self.post_layernorm = LayerNorm(vc.hidden_size, vc.layer_norm_eps, dtype)

    def __call__(self, pixel_values):  # (B, 3, 512, 512) -> (B, 1024, 768)
        return self.post_layernorm(self.encoder(self.embeddings(pixel_values)))


# -------------------------------------------------------------------- connector
class Idefics3SimpleMLP(nnx.Module):
    def __init__(self, cfg: SmolVLMConfig, dtype):
        self.proj = Linear(cfg.vision.hidden_size * cfg.scale_factor ** 2, cfg.text.hidden_size, False, dtype)

    def __call__(self, x):
        return self.proj(x)


class Idefics3Connector(nnx.Module):
    """Pixel shuffle (1024 -> 64 tokens) followed by a projection into the text width."""

    def __init__(self, cfg: SmolVLMConfig, dtype):
        self.modality_projection = Idefics3SimpleMLP(cfg, dtype)
        self.scale_factor = cfg.scale_factor

    def pixel_shuffle(self, x):
        B, seq, embed = x.shape
        h = w = int(seq ** 0.5)
        sf = self.scale_factor
        x = x.reshape(B, h, w // sf, embed * sf).transpose(0, 2, 1, 3)
        x = x.reshape(B, w // sf, h // sf, embed * sf * sf).transpose(0, 2, 1, 3)
        return x.reshape(B, seq // (sf * sf), embed * sf * sf)

    def __call__(self, x):  # (B, 1024, 768) -> (B, 64, 576)
        return self.modality_projection(self.pixel_shuffle(x))


# ------------------------------------------------------------ text model (Llama)
def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class LlamaRotaryEmbedding(nnx.Module):
    def __init__(self, tc: TextConfig):
        exponent = np.arange(0, tc.head_dim, 2, dtype=np.float32) / np.float32(tc.head_dim)
        self.inv_freq = nnx.Variable(jnp.asarray(1.0 / np.power(np.float32(tc.rope_theta), exponent), jnp.float32))

    def __call__(self, position_ids, dtype):
        freqs = position_ids.astype(jnp.float32)[:, :, None] * self.inv_freq[...][None, None, :]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


class LlamaAttention(nnx.Module):
    def __init__(self, tc: TextConfig, dtype):
        d, hd = tc.hidden_size, tc.head_dim
        self.q_proj = Linear(d, tc.num_attention_heads * hd, False, dtype)
        self.k_proj = Linear(d, tc.num_key_value_heads * hd, False, dtype)
        self.v_proj = Linear(d, tc.num_key_value_heads * hd, False, dtype)
        self.o_proj = Linear(tc.num_attention_heads * hd, d, False, dtype)
        self.num_heads, self.num_kv_heads, self.head_dim = tc.num_attention_heads, tc.num_key_value_heads, hd
        self.groups = tc.num_attention_heads // tc.num_key_value_heads

    def __call__(self, x, cos, sin, mask):
        B, S, _ = x.shape
        split = lambda t, n: t.reshape(B, S, n, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(self.q_proj(x), self.num_heads), split(self.k_proj(x), self.num_kv_heads), split(self.v_proj(x), self.num_kv_heads)
        c, s = cos[:, None], sin[:, None]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        k, v = jnp.repeat(k, self.groups, axis=1), jnp.repeat(v, self.groups, axis=1)  # GQA
        out = _sdpa(q, k, v, mask, self.head_dim ** -0.5)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, S, -1))


class LlamaMLP(nnx.Module):
    def __init__(self, tc: TextConfig, dtype):
        self.gate_proj = Linear(tc.hidden_size, tc.intermediate_size, False, dtype)
        self.up_proj = Linear(tc.hidden_size, tc.intermediate_size, False, dtype)
        self.down_proj = Linear(tc.intermediate_size, tc.hidden_size, False, dtype)

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nnx.Module):
    def __init__(self, tc: TextConfig, dtype):
        self.input_layernorm = RMSNorm(tc.hidden_size, tc.rms_norm_eps, dtype)
        self.self_attn = LlamaAttention(tc, dtype)
        self.post_attention_layernorm = RMSNorm(tc.hidden_size, tc.rms_norm_eps, dtype)
        self.mlp = LlamaMLP(tc, dtype)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nnx.Module):
    def __init__(self, tc: TextConfig, dtype):
        self.embed_tokens = Embedding(tc.vocab_size, tc.hidden_size, dtype)
        self.layers = nnx.List([LlamaDecoderLayer(tc, dtype) for _ in range(tc.num_hidden_layers)])
        self.norm = RMSNorm(tc.hidden_size, tc.rms_norm_eps, dtype)
        self.rotary_emb = LlamaRotaryEmbedding(tc)
        self.remat = False

    def __call__(self, inputs_embeds, mask, position_ids):
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)
        h = inputs_embeds
        for layer in self.layers:
            if self.remat:
                graphdef, state = nnx.split(layer)
                h = jax.checkpoint(lambda s, hh: nnx.merge(graphdef, s)(hh, cos, sin, mask))(state, h)
            else:
                h = layer(h, cos, sin, mask)
        return self.norm(h)


# ------------------------------------------------------------------ full model
class Idefics3Model(nnx.Module):
    def __init__(self, cfg: SmolVLMConfig, dtype):
        self.vision_model = Idefics3VisionTransformer(cfg.vision, dtype)
        self.connector = Idefics3Connector(cfg, dtype)
        self.text_model = LlamaModel(cfg.text, dtype)


class SmolVLM(nnx.Module):
    """Teacher-forced forward pass: image tokens in ``input_ids`` are replaced by the 64
    connector features of the image in ``pixel_values``; returns next-token logits."""

    def __init__(self, cfg: SmolVLMConfig, *, dtype=jnp.bfloat16):
        self.cfg = cfg
        self.dtype = dtype
        self.model = Idefics3Model(cfg, dtype)
        self.lm_head = Linear(cfg.text.hidden_size, cfg.text.vocab_size, False, dtype)

    def set_remat(self, enabled: bool) -> None:
        self.model.vision_model.encoder.remat = enabled
        self.model.text_model.remat = enabled

    def _merged_embeds(self, input_ids, pixel_values):
        """Token embeddings with every ``image_token_id`` position replaced, in order, by the
        image features (``pixel_values`` is ``(B, N, 3, H, W)`` with N images per row)."""
        embeds = self.model.text_model.embed_tokens(input_ids)
        pv = pixel_values.astype(self.dtype).reshape((-1,) + pixel_values.shape[2:])
        rows = self.model.connector(self.model.vision_model(pv)).reshape(-1, embeds.shape[-1])
        flat_mask = (input_ids == self.cfg.image_token_id).reshape(-1)
        idx = jnp.clip(jax.lax.associative_scan(jnp.add, flat_mask.astype(jnp.int32)) - 1, 0, rows.shape[0] - 1)
        flat = embeds.reshape(-1, embeds.shape[-1])
        merged = jnp.where(flat_mask[:, None], jnp.take(rows.astype(embeds.dtype), idx, axis=0), flat)
        return merged.reshape(embeds.shape)

    def __call__(self, input_ids, attention_mask, pixel_values, logits_start: int):
        """Logits for positions ``logits_start:`` (the full logits tensor is never built)."""
        embeds = self._merged_embeds(input_ids, pixel_values)
        L = input_ids.shape[1]
        ar = jnp.arange(L)
        mask = (ar[None, None, None, :] <= ar[None, None, :, None]) & attention_mask.astype(bool)[:, None, None, :]
        position_ids = jnp.clip(jax.lax.associative_scan(jnp.add, attention_mask.astype(jnp.int32), axis=-1) - 1, 0)
        h = self.model.text_model(embeds, mask, position_ids)
        return self.lm_head(h[:, logits_start:])

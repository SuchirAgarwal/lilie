#!/usr/bin/env python
"""
Sample a trained JiT-T2I model for the GPIC FD-DINOv2 / PRDC protocol.

python sample_jit.py --config configs/sample/g_pixel+lilie.yaml [key=value ...]
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from common import checkpoint, data, utils
from common.config import load_config, parse_cli, pretty
from jit import diffusion
from jit.config import JiTConfig, SampleConfig
from train_jit import TextEncoder, build_denoiser


def main():
    config_path, overrides = parse_cli(__doc__)
    cfg = load_config(SampleConfig, config_path, overrides)
    utils.init_distributed(cfg.distributed)
    utils.log(pretty(cfg))
    mesh = utils.make_mesh()
    assert cfg.batch_size % mesh.size == 0, "batch_size must be divisible by the device count"

    records = data.read_captions_jsonl(cfg.captions)
    if cfg.max_captions:
        records = records[: cfg.max_captions]
    train_cfg = JiTConfig(**checkpoint.load_meta(cfg.checkpoint)["config"])
    denoiser = build_denoiser(train_cfg, compute_dtype=jnp.float32, remat=False)
    graphdef, params, rest = nnx.split(denoiser, nnx.Param, ...)
    params = checkpoint.load(cfg.checkpoint, "ema_weights" if cfg.use_ema else "train_weights", params)
    model = nnx.merge(graphdef, utils.replicate(mesh, params), utils.replicate(mesh, rest))
    text_encoder = TextEncoder(train_cfg.text_encoder, train_cfg.max_length, mesh)
    size = train_cfg.input_size
    primary = utils.is_primary()

    @jax.jit
    def batch_noise(indices):
        key = jax.random.key(cfg.seed)
        return jax.vmap(lambda i: jax.random.normal(jax.random.fold_in(key, i), (3, size, size), jnp.float32))(indices)

    scales = [float(s) for s in cfg.cfg_scales.replace(",", " ").split()]
    outputs = {scale: np.empty((len(records), size, size, 3), np.uint8) for scale in scales} if primary else {}
    for start in range(0, len(records), cfg.batch_size):
        batch = records[start:start + cfg.batch_size]
        n_valid = len(batch)
        batch = batch + [batch[-1]] * (cfg.batch_size - n_valid)  # pad the last batch
        indices = np.arange(start, start + cfg.batch_size, dtype=np.int32)
        noise = batch_noise(utils.shard_global_batch(mesh, indices))
        condition = text_encoder([r["caption"] for r in batch], global_batch=True)
        for scale in scales:
            samples = diffusion.euler_sample(model, noise, condition, text_encoder.null_condition,
                                             num_steps=cfg.num_steps, guidance=scale, timeshift=train_cfg.timeshift,
                                             t_eps=train_cfg.t_eps, compute_dtype=jnp.float32)
            samples = utils.gather(samples)[:n_valid]  # every process gathers; only process 0 keeps them
            if primary:
                outputs[scale][start:start + n_valid] = np.clip((samples + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        utils.log(f"{start + n_valid}/{len(records)} captions")

    if not primary:
        return
    keys = np.array([r["key"] for r in records])
    captions = np.array([r["caption"] for r in records])
    for scale in scales:
        images = outputs[scale]
        path = f"{cfg.output_dir}/samples_cfg{scale:g}.npz"
        os.makedirs(cfg.output_dir, exist_ok=True)
        np.savez(path, arr_0=images, keys=keys, captions=captions)
        utils.log(f"[write] {path}: {images.shape}")
        if cfg.save_png:
            os.makedirs(f"{cfg.output_dir}/png_cfg{scale:g}", exist_ok=True)
        for i in range(min(cfg.save_png, len(images))):
            Image.fromarray(images[i]).save(f"{cfg.output_dir}/png_cfg{scale:g}/{i:05d}_{keys[i][:12]}.png")


if __name__ == "__main__":
    main()

"""
The frozen LILIE encoder as a loss feature map for JiT-T2I training.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from common import checkpoint
from common.utils import replicate
from lilie.tarflow import Model


def load_frozen_encoder(directory: str, *, use_ema: bool, compute_dtype, remat: bool, mesh):
    """Build the encoder from a released LILIE checkpoint directory and place it on ``mesh``.

    Returns ``(graphdef, state, m2)``; ``state`` is replicated on every device.
    """
    meta = checkpoint.load_meta(directory)
    cfg = meta["config"]
    model = Model(img_size=cfg["image_size"], patch_size=cfg["patch_size"], channels=cfg["channels"],
                  num_blocks=cfg["blocks"], layers_per_block=cfg["layers_per_block"],
                  clip_range=cfg["clip_range"], rngs=nnx.Rngs(0), compute_dtype=compute_dtype, remat=remat)
    graphdef, state = nnx.split(model)
    state = checkpoint.load(directory, "encoder_ema" if use_ema else "encoder", state)
    m2 = np.asarray(jax.device_get(state["m2"][...]))
    assert np.isfinite(m2).all() and (m2 > 0).all() and (m2 == m2.reshape(-1)[0]).all(), \
        "the checkpoint's m2 buffer must hold one positive global second moment"
    return graphdef, replicate(mesh, state), float(m2.reshape(-1)[0])


def make_encode_for_loss(graphdef):
    """``encode(state, pixels)``: NCHW pixels in [-1, 1] -> rescaled NCHW feature image."""

    def encode(state, pixels):
        model = nnx.merge(graphdef, state)
        tokens, _ = model(pixels.astype(jnp.float32))
        tokens = tokens / jnp.sqrt(model.m2[...].astype(jnp.float32))
        return 0.5 * model.unpatchify(tokens)

    return encode

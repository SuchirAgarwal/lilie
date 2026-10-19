"""
DIFFERENTIABLE bicubic resize matching ``torch.nn.functional.interpolate(mode="bicubic",
align_corners=False, antialias=True)``
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


def _cubic_kernel(x: np.ndarray, a: float) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < 1.0, ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0,
                    np.where(x < 2.0, (((x - 5.0) * x + 8.0) * x - 4.0) * a, 0.0))


@functools.lru_cache(maxsize=None)
def _resize_weights(in_size: int, out_size: int) -> np.ndarray:
    """``(out_size, in_size)`` fp32 one-axis resize matrix (torch's antialiased bicubic)."""
    W = np.zeros((out_size, in_size), dtype=np.float64)
    scale = in_size / out_size
    support, invscale = 2.0 * max(scale, 1.0), 1.0 / max(scale, 1.0)
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), in_size) - xmin
        w = _cubic_kernel((np.arange(xsize) + xmin - center + 0.5) * invscale, a=-0.5)
        W[i, xmin:xmin + xsize] = w / w.sum()
    return W.astype(np.float32)

# port of torch antialiased bicubic resize
def resize_bicubic(x: jnp.ndarray, out_h: int, out_w: int) -> jnp.ndarray:
    """``(B, C, H, W)`` fp32 -> ``(B, C, out_h, out_w)`` fp32."""
    hi = jax.lax.Precision.HIGHEST
    wh = jnp.asarray(_resize_weights(x.shape[-2], out_h))
    ww = jnp.asarray(_resize_weights(x.shape[-1], out_w))
    y = jnp.einsum("oh,bchw->bcow", wh, x, precision=hi)
    return jnp.einsum("pw,bcow->bcop", ww, y, precision=hi)

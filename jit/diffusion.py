"""Flow-matching x-prediction loss (JiT) with the optional LILIE term, and the Euler sampler."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _time_image(t):
    return jnp.asarray(t, jnp.float32).reshape((-1, 1, 1, 1))


def time_shift(t, timeshift: float):
    t = jnp.asarray(t, jnp.float32)
    return t / (t + (1.0 - t) * timeshift)


def train_loss(net, x, condition, uncondition, *, key, null_condition_p: float, P_mean: float,
               P_std: float, timeshift: float, t_eps: float, encode_for_loss=None,
               lilie_lambda: float = 0.0, pixel_weight: float = 1.0) -> dict:
    """
    ``w(t) * (pixel_weight * ||x - x_hat||^2 + lambda * ||R(x) - R(x_hat)||^2)`` over the batch.
    """
    x = jnp.asarray(x, jnp.float32)
    b = x.shape[0]
    mask_key, t_key, noise_key = (jax.random.fold_in(key, i) for i in range(3))
    drop = jax.random.uniform(mask_key, (b,)) < null_condition_p
    condition = jnp.where(drop[:, None, None], jnp.broadcast_to(uncondition, condition.shape), condition)
    t = time_shift(jax.nn.sigmoid(jax.random.normal(t_key, (b,)) * P_std + P_mean), timeshift)
    noise = jax.random.normal(noise_key, x.shape, jnp.float32)
    x_t = _time_image(t) * x + (1.0 - _time_image(t)) * noise
    denom = jnp.maximum(1.0 - _time_image(t), t_eps)
    pred = jnp.asarray(net(x_t, t, condition), jnp.float32)
    pixel_error = jnp.square((pred - x) / denom)
    out = {"pixel_loss": pixel_error.mean()}
    if encode_for_loss is None or lilie_lambda == 0.0:
        out["loss"] = pixel_weight * out["pixel_loss"]
        out["loss_per_sample"] = pixel_weight * pixel_error.mean(axis=(1, 2, 3))
        return out
    # One encoder call over the interleaved (pred, x) pairs keeps every row device-local.
    paired = jnp.stack([pred, jax.lax.stop_gradient(x)], axis=1).reshape((2 * b,) + x.shape[1:])
    encoded = encode_for_loss(paired).reshape((b, 2) + x.shape[1:])
    lilie_error = jnp.square((encoded[:, 0] - jax.lax.stop_gradient(encoded[:, 1])) / denom)
    out["lilie_loss"] = lilie_error.mean()
    out["loss"] = pixel_weight * out["pixel_loss"] + lilie_lambda * out["lilie_loss"]
    out["loss_per_sample"] = (pixel_weight * pixel_error + lilie_lambda * lilie_error).mean(axis=(1, 2, 3))
    return out


def euler_timesteps(num_steps: int, timeshift: float = 1.0):
    steps = jnp.concatenate([jnp.linspace(0.0, 1.0 - 1.0 / num_steps, num_steps, dtype=jnp.float32), jnp.ones((1,), jnp.float32)])
    return time_shift(steps, timeshift)


def _euler_sampler(net, noise, condition, uncondition, *, num_steps, guidance, timeshift, t_eps, compute_dtype):
    x = jnp.asarray(noise, jnp.float32)
    b = x.shape[0]
    uncondition = jnp.broadcast_to(uncondition, condition.shape)
    cfg_condition = jnp.concatenate([uncondition, condition], axis=0)
    steps = euler_timesteps(num_steps, timeshift)

    def step(x, times):
        t_cur, t_next = times
        t = jnp.full((2 * b,), t_cur, jnp.float32)
        pred = jnp.asarray(net(jnp.concatenate([x, x]), t, cfg_condition, compute_dtype=compute_dtype), jnp.float32)
        velocity = (pred - jnp.concatenate([x, x])) / jnp.maximum(1.0 - _time_image(t), t_eps)
        uncond, cond = jnp.split(velocity, 2, axis=0)
        scale = jnp.where(t_cur > 0.0, guidance, 1.0)
        return x + (uncond + scale * (cond - uncond)) * (t_next - t_cur), None

    x, _ = jax.lax.scan(step, x, (steps[:-1], steps[1:]))
    return x


_compiled_euler_sampler = jax.jit(_euler_sampler, static_argnames=("num_steps", "compute_dtype"))


def euler_sample(net, noise, condition, uncondition, *, num_steps: int, guidance: float,
                 timeshift: float = 1.0, t_eps: float = 0.05, compute_dtype=jnp.float32):
    """Classifier-free-guided forward Euler from ``noise`` (t=0) to an image (t=1)."""
    return _compiled_euler_sampler(net, noise, condition, uncondition, num_steps=num_steps,
                                   guidance=guidance, timeshift=timeshift, t_eps=t_eps,
                                   compute_dtype=compute_dtype)

#!/usr/bin/env python
"""Train JiT-T2I on GPIC, optionally with the LILIE loss term (Algorithm 2 of the paper).

python train_jit.py --config configs/jit/g_pixel+lilie.yaml [key=value ...]
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image

from common import checkpoint, data, utils
from common.config import load_config, parse_cli, pretty
from jit import diffusion, lilie_loss, qwen3
from jit.config import JiTConfig
from jit.model import JiT_T2I

SAMPLE_CAPTIONS = (
    "People stand in line at an airport terminal near a large digital flight board showing departures.",
    "cat statue, brick base, tiled ground, garden, outdoor.",
    "People stand on the deck of a large warship under a cloudy sky.",
    "Several vintage typewriters and early computer keyboards are displayed on a wooden desk.",
    "motorcycle, rider, sky, forest, jump",
    "A stone bell tower with four visible bells stands under a clear blue sky.",
    "Five men are posing closely together in a group photo at night, all smiling.",
    "A brown tile with a golden lion design and decorative patterns.",
    "A dog sitting on green grass in a park.",
    "A teacup on a table.",
    "A white boat floating on a red lake.",
    "A sunflower under a blue sky.",
)


def build_denoiser(cfg: JiTConfig, *, compute_dtype=jnp.bfloat16, remat: bool | None = None) -> JiT_T2I:
    return JiT_T2I(input_size=cfg.input_size, patch_size=cfg.patch_size, hidden_size=cfg.hidden_size,
                   num_heads=cfg.num_heads, num_blocks=cfg.num_blocks, num_text_blocks=cfg.num_text_blocks,
                   bottleneck_dim=cfg.bottleneck_dim, mlp_ratio=cfg.mlp_ratio, txt_embed_dim=cfg.txt_embed_dim,
                   txt_max_length=cfg.max_length, rngs=nnx.Rngs(cfg.seed), compute_dtype=compute_dtype,
                   remat=cfg.remat if remat is None else remat)


class TextEncoder:
    """Frozen Qwen3 conditioner: captions -> ``(B, max_length, 2048)``"""

    def __init__(self, model_name: str, max_length: int, mesh):
        self.tokenizer = qwen3.load_tokenizer(model_name)
        model, _ = qwen3.build_qwen3(model_name)
        self.graphdef, state = nnx.split(model)
        self.state = utils.replicate(mesh, state)
        self.max_length, self.mesh = max_length, mesh

        @jax.jit
        def forward(state, ids, mask):
            return jax.lax.stop_gradient(nnx.merge(self.graphdef, state)(ids, mask))

        self._forward = forward
        ids, mask = qwen3.tokenize(self.tokenizer, [""], max_length)
        self.null_condition = utils.replicate(mesh, forward(utils.replicate(mesh, state), *utils.replicate(mesh, (ids, mask))))

    def __call__(self, captions, *, global_batch: bool = False):
        """``global_batch=True`` when every process passes the same captions."""
        ids, mask = qwen3.tokenize(self.tokenizer, captions, self.max_length)
        place = utils.shard_global_batch if global_batch else utils.shard_batch
        return self._forward(self.state, *place(self.mesh, (ids, mask)))


# ------------------------------------------------------------------- train step
def make_train_step(cfg: JiTConfig, denoiser, encode_for_loss):
    graphdef, params, rest = nnx.split(denoiser, nnx.Param, ...)
    tx = optax.chain(optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.eps),
                     optax.add_decayed_weights(cfg.weight_decay))

    def loss_fn(params, rest, images, condition, uncondition, key, flow_state):
        model = nnx.merge(graphdef, params, rest)
        encode = None if flow_state is None else (lambda pixels: encode_for_loss(flow_state, pixels))
        return_dict = diffusion.train_loss(
            model, images, condition, uncondition, key=key, null_condition_p=cfg.null_condition_p,
            P_mean=cfg.P_mean, P_std=cfg.P_std, timeshift=cfg.timeshift, t_eps=cfg.t_eps,
            encode_for_loss=encode, lilie_lambda=cfg.lilie_lambda, pixel_weight=cfg.pixel_weight)
        return return_dict["loss"], {k: v for k, v in return_dict.items() if k != "loss_per_sample"}

    @functools.partial(jax.jit, donate_argnums=(0, 2, 3))
    def train_step(params, rest, opt_state, ema, images, condition, uncondition, key, lr, flow_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, rest, images, condition, uncondition, key, flow_state)
        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        grads = jax.tree.map(lambda g: jnp.where(finite, g, 0.0), grads)
        safe_norm = jnp.where(finite, grad_norm, 1.0)
        grads = jax.tree.map(lambda g: jnp.where(safe_norm < cfg.grad_clip, g, g / safe_norm * cfg.grad_clip), grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, jax.tree.map(lambda u: -jnp.where(finite, lr, 0.0) * u, updates))
        ema = utils.update_ema(ema, params, cfg.ema_decay)
        return params, opt_state, ema, aux, grad_norm, finite

    @jax.jit
    def eval_step(params, rest, images, condition, uncondition, key, flow_state):
        return loss_fn(params, rest, images, condition, uncondition, key, flow_state)[1]

    return {"graphdef": graphdef, "params": params, "rest": rest, "tx": tx,
            "train_step": train_step, "eval_step": eval_step}


def image_grid(samples_nchw: np.ndarray, columns: int = 4) -> Image.Image:
    images = np.clip((samples_nchw + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    n, h, w, _ = images.shape
    rows = math.ceil(n / columns)
    grid = np.zeros((rows * h, columns * w, 3), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = image
    return Image.fromarray(grid)


def main():
    config_path, overrides = parse_cli(__doc__)
    cfg = load_config(JiTConfig, config_path, overrides)
    utils.init_distributed(cfg.distributed)
    utils.log(pretty(cfg))
    mesh = utils.make_mesh()
    n_proc = jax.process_count()
    assert cfg.batch_size % (n_proc * jax.local_device_count()) == 0, "batch_size must be divisible by the device count"
    local_batch = cfg.batch_size // n_proc
    root_key = utils.set_seed(cfg.seed)

    denoiser = build_denoiser(cfg)
    utils.log(f"[denoiser] {utils.count_params(nnx.state(denoiser, nnx.Param)):,} parameters")
    text_encoder = TextEncoder(cfg.text_encoder, cfg.max_length, mesh)
    flow_state = encode_for_loss = None
    if cfg.lilie_checkpoint:
        flow_gd, flow_state, m2 = lilie_loss.load_frozen_encoder(
            cfg.lilie_checkpoint, use_ema=cfg.lilie_use_ema, compute_dtype=jnp.bfloat16, remat=cfg.lilie_remat, mesh=mesh)
        encode_for_loss = lilie_loss.make_encode_for_loss(flow_gd)
        utils.log(f"[lilie] {cfg.lilie_checkpoint}: lambda {cfg.lilie_lambda}, pixel weight {cfg.pixel_weight}, m2 {m2:.4f}")

    # ---- resume -------------------------------------------------------------------------
    step = 0
    resume_dir = checkpoint.latest_dir(cfg.run_dir)
    if resume_dir is not None:
        step = checkpoint.load_meta(resume_dir)["step"]
        nnx.update(denoiser, checkpoint.load(resume_dir, "train_weights", nnx.state(denoiser, nnx.Param)))
        utils.log(f"[resume] {resume_dir} at step {step}")
    ts = make_train_step(cfg, denoiser, encode_for_loss)
    params, rest = utils.replicate(mesh, ts["params"]), utils.replicate(mesh, ts["rest"])
    if resume_dir is not None:
        ema = utils.replicate(mesh, checkpoint.load(resume_dir, "ema_weights", ts["params"]))
        opt_state = utils.replicate(mesh, checkpoint.load(resume_dir, "opt", jax.eval_shape(ts["tx"].init, params)))
    else:
        ema = utils.init_ema(params)
        opt_state = jax.jit(ts["tx"].init)(params)

    # ---- data ---------------------------------------------------------------------------
    loader_kw = dict(num_workers=cfg.num_workers, process_index=jax.process_index(), process_count=n_proc,
                     prefetch_factor=cfg.prefetch_factor)
    train_loader = data.build_loader("train", data.JiTTransform(cfg.input_size), local_batch=local_batch,
                                     shuffle_buffer=cfg.shuffle_buffer, resume_step=step, **loader_kw)
    train_iter = iter(train_loader)
    utils.log(f"[data] {cfg.num_workers} loader workers per host are downloading their first GPIC shards; "
              "the first steps also include compilation")
    uncondition = text_encoder.null_condition

    val_loader = data.build_loader("val", data.JiTTransform(cfg.input_size, is_val=True), local_batch=local_batch,
                                   shuffle_buffer=0, infinite=False,
                                   **{**loader_kw, "num_workers": min(cfg.num_workers, 2)})

    def run_validation():
        totals, n = {}, 0
        for i, (images, captions, _) in enumerate(utils.synchronized_batches(iter(val_loader), cfg.val_num_images // cfg.batch_size)):
            key = jax.random.fold_in(root_key, 1_000_000_000 + i)
            aux = ts["eval_step"](params, rest, utils.shard_batch(mesh, images), text_encoder(captions),
                                  uncondition, key, flow_state)
            for k, v in aux.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n += 1
        return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}

    def sample_grid():
        n_pad = -len(SAMPLE_CAPTIONS) % mesh.size
        captions = list(SAMPLE_CAPTIONS) + [SAMPLE_CAPTIONS[-1]] * n_pad
        condition = text_encoder(captions, global_batch=True)
        model = nnx.merge(ts["graphdef"], ema, rest)
        noise = jax.jit(lambda k: jax.random.normal(k, (len(captions), 3, cfg.input_size, cfg.input_size)),
                        out_shardings=utils.batch_sharding(mesh))(jax.random.fold_in(root_key, 2_000_000_000 + step))
        samples = diffusion.euler_sample(model, noise, condition, uncondition, num_steps=50, guidance=cfg.sample_cfg,
                                         timeshift=cfg.timeshift, t_eps=cfg.t_eps, compute_dtype=jnp.float32)
        samples = utils.gather(samples)[: len(SAMPLE_CAPTIONS)]
        if utils.is_primary():
            os.makedirs(f"{cfg.run_dir}/samples", exist_ok=True)
            image_grid(samples).save(f"{cfg.run_dir}/samples/step-{step:09d}.png")

    def save(directory_fn, *, rolling: bool):
        trees = {"train_weights": params, "ema_weights": ema}
        if rolling:
            trees["opt"] = opt_state
        meta = {"step": step, "config": dataclasses.asdict(cfg)}
        utils.log(f"[save] {directory_fn(cfg.run_dir, step, trees, meta)}")

    wandb = None
    if cfg.wandb_project and utils.is_primary():
        import wandb
        name = os.path.basename(cfg.run_dir.rstrip("/"))
        wandb.init(project=cfg.wandb_project, name=name, id=name, resume="allow", config=dataclasses.asdict(cfg))

    # ---- train loop ---------------------------------------------------------------------
    milestones = checkpoint.parse_steps(cfg.milestone_steps)
    t_last = time.time()
    while step < cfg.total_steps:
        lr = cfg.lr * utils.warmup_flat_cooldown(step, cfg.warmup_steps, cfg.cooldown_steps, cfg.total_steps)
        images, captions, _ = next(train_iter)
        condition = text_encoder(captions)
        key = jax.random.fold_in(root_key, step)
        params, opt_state, ema, aux, grad_norm, finite = ts["train_step"](
            params, rest, opt_state, ema, utils.shard_batch(mesh, images), condition, uncondition, key, lr, flow_state)
        step += 1
        if not bool(utils.local_replica(finite)):
            utils.log(f"[warn] non-finite loss or gradient at step {step}; step skipped")

        if step % cfg.log_every == 0:
            now = time.time()
            metrics = {f"train/{k}": float(v) for k, v in aux.items()}
            metrics.update({"train/grad_norm": float(grad_norm), "train/lr": lr,
                            "train/steps_per_sec": cfg.log_every / (now - t_last)})
            t_last = now
            utils.log(f"step {step:>7d}  loss {metrics['train/loss']:.4f}  gnorm {metrics['train/grad_norm']:.3f}  "
                      f"lr {lr:.2e}  {metrics['train/steps_per_sec']:.2f} it/s")
            if wandb:
                wandb.log(metrics, step=step)
        if step % cfg.val_every == 0:
            metrics = run_validation()
            utils.log(f"[val] step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            if wandb:
                wandb.log(metrics, step=step)
        if cfg.sample_every and step % cfg.sample_every == 0:
            sample_grid()
        if step in milestones:
            save(checkpoint.save_milestone, rolling=False)
        if step % cfg.save_every == 0 or step == cfg.total_steps:
            save(checkpoint.save_rolling, rolling=True)
            t_last = time.time()
    utils.log("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train the LILIE encoder (Algorithm 1 of the paper).

    python train_lilie.py --config configs/lilie/small.yaml [key=value ...]
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from common import checkpoint, data, hf, utils
from common.config import load_config, parse_cli, pretty
from lilie import image_ops, lora, smolvlm, tarflow
from lilie.captions import CaptionIO
from lilie.config import LILIEConfig

VLM_IMAGE_SIZE = 512  # SmolVLM's native tile; the 256 px feature image is upsampled to it


def build_encoder(cfg: LILIEConfig, seed: int) -> tarflow.Model:
    return tarflow.Model(img_size=cfg.image_size, patch_size=cfg.patch_size, channels=cfg.channels,
                         num_blocks=cfg.blocks, layers_per_block=cfg.layers_per_block,
                         clip_range=cfg.clip_range, rngs=nnx.Rngs(seed), compute_dtype=jnp.float32,
                         remat=cfg.flow_remat)


def build_vlm(cfg: LILIEConfig) -> tuple[smolvlm.SmolVLM, str]:
    """Pretrained SmolVLM (bf16, frozen) with LoRA adapters on every linear layer but the head."""
    snapshot = hf.snapshot(cfg.vlm_name, cfg.vlm_revision)
    with open(os.path.join(snapshot, "config.json")) as handle:
        model = smolvlm.SmolVLM(smolvlm.config_from_hf(json.load(handle)), dtype=jnp.bfloat16)
    hf.load_hf_state(model, hf.load_snapshot_tensors(snapshot), jnp.bfloat16)
    n = lora.inject_lora(model, cfg.lora_rank, cfg.lora_alpha, rngs=nnx.Rngs(cfg.seed + 1))
    utils.log(f"[vlm] {cfg.vlm_name}: LoRA rank {cfg.lora_rank} on {n} linear layers")
    model.set_remat(cfg.vlm_remat)
    return model, snapshot


def encoder_config(cfg: LILIEConfig) -> dict:
    """The fields a consumer needs to rebuild the encoder (saved in every checkpoint)."""
    return {k: getattr(cfg, k) for k in ("image_size", "patch_size", "channels", "blocks", "layers_per_block", "clip_range")}


def half_rms(t, target: float, eps: float = 1e-3):
    """Rescale every image in the batch to RMS ``target``."""
    n = jnp.linalg.norm(t.reshape(t.shape[0], -1), axis=1).reshape(-1, 1, 1, 1)
    d = math.prod(t.shape[1:])
    return t * (target * d ** 0.5) / (n + eps)


def student_inputs(encoder, x, cfg: LILIEConfig, noise, sigma: float):
    """Image -> encoder -> RMS-1/2 feature image -> + sigma * noise -> VLM pixel input."""
    z, _ = encoder(x)
    batch_m2 = jax.lax.stop_gradient(jnp.mean(jnp.square(z)))  # global E[z^2] of this batch
    f = half_rms(encoder.unpatchify(z), cfg.feature_rms)
    if sigma > 0:
        f = f + sigma * noise
    f = image_ops.resize_bicubic(f, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE)
    return f[:, None].astype(jnp.bfloat16), batch_m2  # (B, 1 image, 3, 512, 512)


def caption_nll(vlm, tok, pixel_values, plen: int):
    """Summed caption NLL and the number of caption tokens (loss = nll / tokens)."""
    logits = vlm(tok["input_ids"], tok["attention_mask"], pixel_values, logits_start=plen - 1)
    logits = logits[:, :-1].astype(jnp.float32)  # position t predicts token t+1
    target = tok["labels"][:, plen:]
    keep = target != -100
    logp = jax.nn.log_softmax(logits, axis=-1)
    ll = jnp.take_along_axis(logp, jnp.where(keep, target, 0)[..., None], axis=-1)[..., 0]
    return -(ll * keep).sum(), keep.sum()


@functools.partial(jax.jit, static_argnames=("initialized",))
def update_m2(m2, observation, decay: float, *, initialized: bool):
    """EMA of the encoder's global second moment, broadcast into the ``m2`` buffer."""
    candidate = decay * m2.reshape(-1)[0] + (1.0 - decay) * observation if initialized else observation
    ok = jnp.isfinite(candidate) & (candidate > 0)
    return jnp.where(ok, jnp.full_like(m2, candidate), m2), ok


def make_train_step(cfg: LILIEConfig, encoder, vlm, plen: int):
    flow_gd, flow_params, flow_rest = nnx.split(encoder, nnx.Param, ...)
    vlm_gd, vlm_params, vlm_rest = nnx.split(vlm, lora.LoRAParam, ...)
    vlm_scale = cfg.vlm_lr / cfg.lr

    def adamw(weight_decay):
        return optax.chain(optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=1e-8),
                           optax.add_decayed_weights(weight_decay))

    tx = optax.multi_transform(
        {"flow": adamw(cfg.weight_decay), "vlm": adamw(0.0)},
        lambda params: {"flow": jax.tree.map(lambda _: "flow", params["flow"]),
                        "vlm": jax.tree.map(lambda _: "vlm", params["vlm"])})

    def loss_fn(params, flow_rest, vlm_rest, batch, noise, sigma):
        encoder = nnx.merge(flow_gd, params["flow"], flow_rest)
        vlm = nnx.merge(vlm_gd, params["vlm"], vlm_rest)
        pixel_values, batch_m2 = student_inputs(encoder, batch["image"], cfg, noise, sigma)
        nll, tokens = caption_nll(vlm, batch, pixel_values, plen)
        return nll / tokens, batch_m2

    @jax.jit
    def grad_step(params, flow_rest, vlm_rest, batch, key):
        noise = jax.random.normal(key, batch["image"].shape, jnp.float32)
        (loss, batch_m2), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, flow_rest, vlm_rest, batch, noise, cfg.sigma)
        return loss, batch_m2, grads, optax.global_norm(grads)

    @functools.partial(jax.jit, donate_argnums=(0, 2))
    def apply_step(grads, params, opt_state, grad_norm, clip_threshold, lr):
        coef = jnp.minimum(1.0, clip_threshold / (grad_norm + 1e-6))
        grads = jax.tree.map(lambda g: coef * g, grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        updates = {"flow": jax.tree.map(lambda u: -lr * u, updates["flow"]),
                   "vlm": jax.tree.map(lambda u: -lr * vlm_scale * u, updates["vlm"])}
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def val_step(params, flow_rest, vlm_rest, batch):
        """Clean (sigma = 0) caption NLL on a validation batch."""
        encoder = nnx.merge(flow_gd, params["flow"], flow_rest)
        vlm = nnx.merge(vlm_gd, params["vlm"], vlm_rest)
        pixel_values, _ = student_inputs(encoder, batch["image"], cfg, None, 0.0)
        return caption_nll(vlm, batch, pixel_values, plen)

    ema_step = jax.jit(functools.partial(utils.update_ema, decay=cfg.ema_decay), donate_argnums=(0,))
    return {"flow_params": flow_params, "flow_rest": flow_rest, "vlm_params": vlm_params, "vlm_rest": vlm_rest,
            "tx": tx, "grad_step": grad_step, "apply_step": apply_step, "val_step": val_step, "ema_step": ema_step}


def main():
    config_path, overrides = parse_cli(__doc__)
    cfg = load_config(LILIEConfig, config_path, overrides)
    utils.init_distributed(cfg.distributed)
    utils.log(pretty(cfg))
    mesh = utils.make_mesh()
    n_proc, n_dev = jax.process_count(), jax.local_device_count()
    assert cfg.batch_size % (n_proc * n_dev) == 0, "batch_size must be divisible by the device count"
    local_batch = cfg.batch_size // n_proc
    root_key = utils.set_seed(cfg.seed)

    encoder = build_encoder(cfg, cfg.seed)
    vlm, vlm_dir = build_vlm(cfg)
    caption_io = CaptionIO(vlm_dir, cfg.image_size, cfg.max_caption_tokens)
    utils.log(f"[encoder] {utils.count_params(nnx.state(encoder, nnx.Param)):,} parameters; "
              f"prompt length {caption_io.plen}, sequence width {caption_io.width}")

    # ---- resume -------------------------------------------------------------------------
    step, m2_initialized = 0, False
    autoclip = utils.AutoClip(cfg.autoclip_percentile, cfg.total_steps)
    resume_dir = checkpoint.latest_dir(cfg.run_dir)
    if resume_dir is not None:
        meta = checkpoint.load_meta(resume_dir)
        step, m2_initialized = meta["step"], meta["m2_initialized"]
        nnx.update(encoder, checkpoint.load(resume_dir, "encoder", nnx.state(encoder)))
        nnx.update(vlm, checkpoint.load(resume_dir, "vlm_lora", nnx.state(vlm, lora.LoRAParam)))
        autoclip.load(checkpoint.load_arrays(resume_dir, "autoclip"))
        utils.log(f"[resume] {resume_dir} at step {step}")

    ts = make_train_step(cfg, encoder, vlm, caption_io.plen)
    params = utils.replicate(mesh, {"flow": ts["flow_params"], "vlm": ts["vlm_params"]})
    flow_rest, vlm_rest = utils.replicate(mesh, ts["flow_rest"]), utils.replicate(mesh, ts["vlm_rest"])
    if resume_dir is not None:
        # encoder_ema is saved with the full encoder state's keys; only its Param leaves are the EMA.
        ema_full = checkpoint.load(resume_dir, "encoder_ema", nnx.State.merge(ts["flow_params"], ts["flow_rest"]))
        ema_params = utils.replicate(mesh, ema_full.filter(nnx.Param))
        opt_state = utils.replicate(mesh, checkpoint.load(resume_dir, "opt", jax.eval_shape(ts["tx"].init, params)))
    else:
        ema_params = utils.init_ema(params["flow"])
        opt_state = jax.jit(ts["tx"].init)(params)

    # ---- data ---------------------------------------------------------------------------
    loader_kw = dict(num_workers=cfg.num_workers, process_index=jax.process_index(), process_count=n_proc)
    train_loader = data.build_loader("train", data.LILIETransform(cfg.image_size), local_batch=local_batch,
                                     shuffle_buffer=cfg.shuffle_buffer, resume_step=step, **loader_kw)
    train_iter = iter(train_loader)
    utils.log(f"[data] {cfg.num_workers} loader workers per host are downloading their first GPIC shards; "
              "the first steps also include compilation")

    def make_batch(images, captions, caption_types):
        tok = caption_io.collate(captions, caption_types)
        return utils.shard_batch(mesh, {"image": images, **tok})

    val_loader = data.build_loader("val", data.LILIETransform(cfg.image_size), local_batch=local_batch,
                                   shuffle_buffer=0, infinite=False,
                                   **{**loader_kw, "num_workers": min(cfg.num_workers, 2)})  # a few batches only

    def run_validation():
        nll = tokens = 0.0
        for images, captions, caption_types in utils.synchronized_batches(iter(val_loader), cfg.val_num_images // cfg.batch_size):
            n, t = ts["val_step"](params, flow_rest, vlm_rest, make_batch(images, captions, caption_types))
            nll, tokens = nll + float(n), tokens + float(t)
        return nll / max(tokens, 1.0)

    def save(directory_fn, *, rolling: bool):
        """Write encoder / encoder_ema (full encoder state incl. ``m2``), vlm_lora, and for rolling saves opt."""
        trees = {"encoder": nnx.State.merge(params["flow"], flow_rest),
                 "encoder_ema": nnx.State.merge(ema_params, flow_rest), "vlm_lora": params["vlm"]}
        meta = {"step": step, "config": encoder_config(cfg),
                "lora_rank": cfg.lora_rank, "lora_alpha": cfg.lora_alpha, "train_config": dataclasses.asdict(cfg)}
        if rolling:
            trees.update(opt=opt_state, autoclip=autoclip.state())
            meta["m2_initialized"] = m2_initialized
        utils.log(f"[save] {directory_fn(cfg.run_dir, step, trees, meta)}")

    wandb = None
    if cfg.wandb_project and utils.is_primary():
        import wandb
        wandb.init(project=cfg.wandb_project, name=os.path.basename(cfg.run_dir.rstrip("/")),
                   config=dataclasses.asdict(cfg), resume="allow", id=os.path.basename(cfg.run_dir.rstrip("/")))

    # ---- train loop ---------------------------------------------------------------------
    milestones = checkpoint.parse_steps(cfg.milestone_steps)
    t_last = time.time()
    while step < cfg.total_steps:
        lr = utils.cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.min_lr, cfg.lr)
        batch = make_batch(*next(train_iter))
        key = jax.random.fold_in(root_key, step)
        loss, batch_m2, grads, grad_norm = ts["grad_step"](params, flow_rest, vlm_rest, batch, key)
        grad_norm_v = float(grad_norm)
        if math.isfinite(grad_norm_v):
            threshold = autoclip.observe(grad_norm_v)
            params, opt_state = ts["apply_step"](grads, params, opt_state, grad_norm, threshold, lr)
            new_m2, ok = update_m2(flow_rest["m2"][...], batch_m2, cfg.latent_m2_decay, initialized=m2_initialized)
            if bool(utils.local_replica(ok)):
                flow_rest["m2"][...] = new_m2
                m2_initialized = True
            ema_params = ts["ema_step"](ema_params, params["flow"])
        else:
            utils.log(f"[warn] non-finite gradient norm at step {step}; step skipped")
        del grads
        step += 1

        if step % cfg.log_every == 0:
            now = time.time()
            metrics = {"train/loss": float(loss), "train/grad_norm": grad_norm_v, "train/lr": lr,
                       "train/latent_rms": math.sqrt(float(utils.local_replica(batch_m2))),
                       "train/steps_per_sec": cfg.log_every / (now - t_last)}
            t_last = now
            utils.log(f"step {step:>7d}  loss {metrics['train/loss']:.4f}  gnorm {grad_norm_v:.3f}  "
                      f"lr {lr:.2e}  latent rms {metrics['train/latent_rms']:.3f}  {metrics['train/steps_per_sec']:.2f} it/s")
            if wandb:
                wandb.log(metrics, step=step)
        if step % cfg.val_every == 0:
            val_loss = run_validation()
            utils.log(f"[val] step {step}: caption loss {val_loss:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)
        if step in milestones:
            save(checkpoint.save_milestone, rolling=False)
        if step % cfg.save_every == 0 or step == cfg.total_steps:
            save(checkpoint.save_rolling, rolling=True)
            t_last = time.time()
    utils.log("done")


if __name__ == "__main__":
    main()

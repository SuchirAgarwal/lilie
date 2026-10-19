"""JiT-T2I training / sampling config (see ``configs/jit/*.yaml`` and ``configs/sample/*.yaml``)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JiTConfig:
    # denoiser (JiT-T2I-g by default; XL is hidden 1024 / heads 16 / blocks 24 / bottleneck 128)
    input_size: int = 256
    patch_size: int = 16
    hidden_size: int = 1536
    num_heads: int = 24
    num_blocks: int = 16
    num_text_blocks: int = 4
    bottleneck_dim: int = 256
    mlp_ratio: float = 4.0
    remat: bool = True
    # text encoder
    text_encoder: str = "Qwen/Qwen3-1.7B"
    max_length: int = 300
    txt_embed_dim: int = 2048
    # flow matching
    P_mean: float = -0.8
    P_std: float = 0.8
    timeshift: float = 1.0
    t_eps: float = 0.05
    null_condition_p: float = 0.1

    lilie_checkpoint: str = ""       # released LILIE checkpoint directory; empty = pixel loss only
    lilie_lambda: float = 0.0
    pixel_weight: float = 1.0       # 0 with lilie_lambda 1 is the paper's "pure LILIE" objective
    lilie_use_ema: bool = False      # score with the EMA encoder weights instead of the trained ones
    lilie_remat: bool = True
    # optimization
    batch_size: int = 256           # global batch
    lr: float = 1e-4
    warmup_steps: int = 4000
    cooldown_steps: int = 0
    total_steps: int = 390_600
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    seed: int = 0
    # data
    num_workers: int = 8
    prefetch_factor: int = 8
    shuffle_buffer: int = 100
    # logging / checkpoints
    run_dir: str = "runs/jit-g"
    log_every: int = 100
    val_every: int = 3906
    val_num_images: int = 4096
    sample_every: int = 5000        # PNG grid of 12 fixed captions (0 = off)
    sample_cfg: float = 6.25
    save_every: int = 500
    milestone_steps: str = "39060, 100000, 200000, 300000, 390600"
    wandb_project: str = ""
    distributed: bool = False

    def validate(self):
        assert self.input_size % self.patch_size == 0 and self.hidden_size % self.num_heads == 0
        assert (self.lilie_lambda > 0) == bool(self.lilie_checkpoint), \
            "set lilie_checkpoint together with a positive lilie_lambda"
        assert self.pixel_weight >= 0 and self.pixel_weight + self.lilie_lambda > 0, "the loss needs a positive term"


@dataclass
class SampleConfig:
    checkpoint: str = ""            # milestone directory holding train_weights.safetensors / ema_weights.safetensors
    captions: str = ""              # JSONL with one {"key", "caption", "caption_type"} per line
    output_dir: str = "samples"
    cfg_scales: str = "5.5, 6.25, 7, 7.75, 8.5"
    num_steps: int = 50
    batch_size: int = 128           # global batch
    use_ema: bool = True
    seed: int = 0
    max_captions: int = 0           # 0 = all lines
    save_png: int = 0               # also write the first N samples as PNGs (0 = none)
    distributed: bool = False

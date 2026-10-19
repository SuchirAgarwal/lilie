"""LILIE training config (see ``configs/lilie/*.yaml``)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LILIEConfig:
    # encoder (improved TARFlow)
    image_size: int = 256
    patch_size: int = 16
    channels: int = 384            # Transformer width inside each coupling block
    blocks: int = 4                # autoregressive coupling blocks
    layers_per_block: int = 4      # Transformer layers per block
    clip_range: float = 1.0        # clip on the predicted log-scale and shift
    flow_remat: bool = True        # recompute blocks in backward (memory)
    # objective
    sigma: float = 0.25            # std of the Gaussian noise added to the RMS-1/2 features
    feature_rms: float = 0.5       # RMS target of the feature image shown to the VLM
    latent_m2_decay: float = 0.99  # EMA of E[z^2] stored in the checkpoint for generator training
    # VLM
    vlm_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    vlm_revision: str = "7e3e67edbbed1bf9888184d9df282b700a323964"
    lora_rank: int = 64
    lora_alpha: float = 32.0
    max_caption_tokens: int = 256
    vlm_remat: bool = True
    # optimisation
    batch_size: int = 256          # global batch
    lr: float = 4e-4               # encoder peak learning rate
    vlm_lr: float = 4e-4           # LoRA peak learning rate
    min_lr: float = 1e-6
    warmup_steps: int = 20_000
    total_steps: int = 390_600     # one epoch of GPIC at batch 256
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-4     # encoder only (LoRA adapters are not decayed)
    autoclip_percentile: float = 10.0
    ema_decay: float = 0.9999      # spectator EMA of the encoder weights
    seed: int = 100
    # data
    num_workers: int = 8
    shuffle_buffer: int = 1000
    # logging / checkpoints
    run_dir: str = "runs/lilie-small"
    log_every: int = 20
    val_every: int = 3906
    val_num_images: int = 4096
    save_every: int = 500
    milestone_steps: str = "3906, 39060, 100000, 200000, 300000, 390600"
    wandb_project: str = ""        # empty = no W&B logging
    distributed: bool = False      # multi-host (TPU pod) run

    def validate(self):
        assert self.image_size % self.patch_size == 0
        assert self.channels % 64 == 0, "heads are channels // 64"
        assert self.lora_rank > 0 and 0 < self.autoclip_percentile <= 100

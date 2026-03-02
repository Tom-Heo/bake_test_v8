from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Data
    data_dir: str = "dataset"
    patch_size: int = 512

    # Training
    epochs: int = 1000
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Model
    bottleneck_dim: int = 1024

    # Augmentation
    augment_strength: float = 0.12

    # Scheduler
    scheduler_gamma: float = 0.999996
    warmup_epochs: int = 10

    # EMA
    ema_decay: float = 0.999

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    max_keep: int = 5

    # Output
    output_dir: str = "outputs"
    log_dir: str = "logs"

from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str
    train_imgs: str
    train_masks: str
    val_imgs: str
    val_masks: str
    num_classes: int
    resolution: int
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float = 0.0
    loss_type: str = "ce"
    binary: bool = False
    save_dir: str = "checkpoints"
    use_cosine: bool = True

    resume: bool = True  # auto-load checkpoint if found in save_dir
    finetune: bool = False  # fine-tuning mode (enables sensible defaults)
    lr_mult: float = 1.0  # scales lr => effective_lr = lr * lr_mult
    use_plateau: bool = False  # alt scheduler; if True, we call sched.step(val_loss) per epoch
    clip_grad: bool = False  # enable gradient clipping
    clip_max_norm: float = 1.0  # max grad norm when clip_grad=True
    checkpoint_path: str = ""
    # (optional but handy)
    seed: int = 42
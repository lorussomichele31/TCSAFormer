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
import os
from pathlib import Path

from TrainConfig import TrainConfig

DATA_DIR = Path(__file__).parent / "datasets"

PRESETS = {
    "isic2018": dict(
        train_imgs = DATA_DIR / "ISIC2018/images/train",
        train_masks= DATA_DIR / "ISIC2018/masks/train",
        val_imgs   = DATA_DIR / "ISIC2018/images/val",
        val_masks  = DATA_DIR / "ISIC2018/masks/val",
        save_dir   = Path("checkpoints") / "isic2018",

        dataset="isic2018",
        num_classes=2, resolution=256, epochs=200, batch_size=8,
        lr=1e-4, optimizer="adam", weight_decay=0.0,
        loss_type="ce", binary=True, use_cosine=True,
        seed=42,
    ),

    "cvc_clinicdb": dict(
        train_imgs = DATA_DIR / "CVC-ClinicDB/images/train",
        train_masks= DATA_DIR / "CVC-ClinicDB/masks/train",
        val_imgs   = DATA_DIR / "CVC-ClinicDB/images/val",
        val_masks  = DATA_DIR / "CVC-ClinicDB/masks/val",
        save_dir   = Path("checkpoints") / "cvc_clinicdb",

        dataset="cvc_clinicdb",
        num_classes=2, resolution=256, epochs=200, batch_size=8,
        lr=1e-4, optimizer="adam", weight_decay=0.0,
        loss_type="ce", binary=True, use_cosine=True,
        seed=42
    ),

    "cvc_clinicdb-finetune": dict(
        # same data
        train_imgs=DATA_DIR / "CVC-ClinicDB/images/train",
        train_masks=DATA_DIR / "CVC-ClinicDB/masks/train",
        val_imgs=DATA_DIR / "CVC-ClinicDB/images/val",
        val_masks=DATA_DIR / "CVC-ClinicDB/masks/val",

        # save to a separate folder to avoid overwriting base run
        save_dir=Path("checkpoints") / "cvc_clinicdb_ft320x10",

        # base meta
        dataset="cvc_clinicdb",
        num_classes=2,
        # finetune tweaks
        resolution=256,
        epochs=10,  # short burst
        batch_size=8,  # adjust if VRAM tight
        lr=1e-4,  # base lr
        optimizer="adamw",  # AdamW tends to work better for FT
        weight_decay=0.0,
        loss_type="ce",
        binary=True,

        # schedulers / knobs
        use_cosine=False,  # disable cosine for FT
        use_plateau=True,  # ReduceLROnPlateau on val loss
        lr_mult=0.1,  # effective lr = lr * lr_mult (=1e-5)
        clip_grad=True,  # stabilize updates
        clip_max_norm=1.0,
        checkpoint_path="checkpoints/cvc_clinicdb/cvc_clinicdb_best.pt",
        # housekeeping
        resume=True,  # auto-load existing ckpt if found
        finetune=True,  # switch on FT behavior
        seed=42,
    ),

    "synapse": dict(
        train_imgs = DATA_DIR / "Synapse/images/train",
        train_masks= DATA_DIR / "Synapse/masks/train",
        val_imgs   = DATA_DIR / "Synapse/images/val",
        val_masks  = DATA_DIR / "Synapse/masks/val",
        save_dir   = Path("checkpoints") / "synapse",

        dataset="synapse",
        num_classes=9,  # 8 organs + background
        resolution=224, epochs=400, batch_size=8,
        lr=1e-4, optimizer="adamw", weight_decay=1e-4,
        loss_type="ce_dice", binary=False, use_cosine=True,
        seed=42,
    ),
}
# -----------------------------------------------------
def build_cfg(name: str) -> TrainConfig:
    p = PRESETS[name]
    return TrainConfig(
        dataset=p["dataset"],
        train_imgs=str(p["train_imgs"]),
        train_masks=str(p["train_masks"]),
        val_imgs=str(p["val_imgs"]),
        val_masks=str(p["val_masks"]),
        num_classes=p["num_classes"],
        resolution=p["resolution"],
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        lr=p["lr"],
        optimizer=p["optimizer"],
        weight_decay=p["weight_decay"],
        loss_type=p["loss_type"],
        binary=p["binary"],
        save_dir=str(p["save_dir"]),
        use_cosine=p["use_cosine"],
        resume=p["resume"],
        finetune=p["finetune"],
        lr_mult=p["lr_mult"],
        use_plateau=p["use_plateau"],
        clip_grad=p["clip_grad"],
        clip_max_norm=p["clip_max_norm"],
        checkpoint_path=p["checkpoint_path"],
        seed=p["seed"],
    )
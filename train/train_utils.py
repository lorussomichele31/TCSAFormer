import glob
import os, math, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from TCSAFormer.TCSAFormer import TCSAFormer
from train.TrainConfig import TrainConfig
from train.logging_utils import dice_per_class, TBLogger


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets, num_classes):
        # logits: (N, C, H, W); targets: (N, H, W) with class indices
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

# Simple Cutout / Random Erasing wrapper
def random_erasing():
    # torchvision RandomErasing applies on tensor (C,H,W) in [0,1]
    return transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')

class SegPairFolder(Dataset):
    """
    Generic 2D image/mask folder dataset.
    Assumes images_dir contains images and masks_dir contains corresponding masks with same stem.
    Masks are expected as single-channel PNG where pixel values are class indices (0..C-1).
    """
    def __init__(self, images_dir, masks_dir, resize, augment=False, binary=False):
        super().__init__()
        self.images = sorted([p for p in Path(images_dir).glob('*') if p.is_file()])
        self.masks_dir = Path(masks_dir)
        self.resize = resize
        self.augment = augment
        self.binary = binary

        # Augmentations: random flip, rotation, Cutout (RandomErasing)
        aug = []
        if augment:
            aug.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
            ])
        self.img_tf = transforms.Compose([
            transforms.Resize(self.resize, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ] + ([random_erasing()] if augment else []))
        self.mask_tf = transforms.Compose([
            transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        x = self.img_tf(img)
        m = torch.from_numpy(np.array(self.mask_tf(mask))).long()

        if self.binary:
            m = (m > 0).long()

        return x, m

def build_dataloaders(cfg: TrainConfig, augment_train=True):
    train_ds = SegPairFolder(cfg.train_imgs, cfg.train_masks, (cfg.resolution, cfg.resolution),
                             augment=augment_train, binary=cfg.binary)
    val_ds   = SegPairFolder(cfg.val_imgs, cfg.val_masks, (cfg.resolution, cfg.resolution),
                             augment=False, binary=cfg.binary)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

def build_model(cfg: TrainConfig, device):
    model = TCSAFormer(num_classes=cfg.num_classes)
    model.to(device)
    return model

def build_optim_sched(cfg: TrainConfig, model, steps_per_epoch: int):
    # Effective LR with multiplier or finetune default
    lr_mult = cfg.lr_mult
    if cfg.finetune and cfg.lr_mult == 1.0:
        lr_mult = 0.1  # sensible default for FT unless user set something else
    eff_lr = cfg.lr * lr_mult

    if cfg.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(),
                                lr=eff_lr, weight_decay=cfg.weight_decay, fused=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=eff_lr)

    sched = None
    if cfg.use_plateau:
        # step once per epoch with validation metric (we’ll call sched.step(val_loss))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3, min_lr=eff_lr*0.1
        )
    elif cfg.use_cosine:
        total_steps = cfg.epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    return opt, sched

def compute_metrics(logits, targets, num_classes, binary=False):
    with torch.no_grad():
        if binary:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).float().mean().item()

        dice_scores = []
        for c in range(num_classes):
            pred_c = (preds == c).float()
            targ_c = (targets == c).float()
            inter = (pred_c * targ_c).sum()
            union = pred_c.sum() + targ_c.sum()
            dice = (2 * inter + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.item())
        mean_dice = float(np.mean(dice_scores))
    return {"acc": correct, "dice": mean_dice}

def loss_fn(cfg: TrainConfig, logits, targets):
    if cfg.loss_type == "ce":
        return F.cross_entropy(logits, targets)
    elif cfg.loss_type == "ce_dice":
        ce = F.cross_entropy(logits, targets)
        dl = DiceLoss()(logits, targets, num_classes=cfg.num_classes)
        return 0.5 * ce + 0.5 * dl
    else:
        raise ValueError(f"Unknown loss_type {cfg.loss_type}")


def load_weights_into(model, ckpt_path: str, device: str = "cuda"):
    import torch
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[resume] loaded checkpoint: {ckpt_path}")
    if missing:    print("[resume] missing keys:", missing)
    if unexpected: print("[resume] unexpected keys:", unexpected)
    return int(ckpt.get("epoch", 0))

def train_one_dataset(cfg: TrainConfig, device):
    # Build data + model
    train_loader, val_loader = build_dataloaders(cfg, augment_train=True)
    model = build_model(cfg, device).to(device)

    # Auto-resume if requested and something exists
    start_epoch = 0
    if cfg.resume:
        ckpt_path = cfg.checkpoint_path
        if ckpt_path:
            start_epoch = load_weights_into(model, ckpt_path, device=device)

    if cfg.finetune:
        # Example: turn on clipping automatically for FT
        if not cfg.clip_grad:
            cfg.clip_grad = True

        # Example: prefer plateau for FT unless user set it
        if not cfg.use_plateau and not cfg.use_cosine:
            cfg.use_plateau = True

    opt, sched = build_optim_sched(cfg, model, steps_per_epoch=len(train_loader))

    # AMP policy
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = amp.GradScaler("cuda", enabled=not use_bf16)

    # Logging
    tb_dir = os.path.join(cfg.save_dir, "tb")
    logger = TBLogger(tb_dir)

    best_val_dice = -1.0 if start_epoch == 0 else float("-inf")
    global_step = 0

    for epoch in range(start_epoch + 1, start_epoch + cfg.epochs + 1):
        model.train()
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                logits = model(x)
                loss = loss_fn(cfg, logits, y)

            if use_bf16:
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
                opt.step()
            else:
                scaler.scale(loss).backward()
                if cfg.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
                scaler.step(opt)
                scaler.update()

            if sched is not None and not cfg.use_plateau:
                # step-per-batch schedulers (e.g., cosine with total_steps)
                sched.step()

            logger.log_scalar("train/loss", loss, global_step)
            logger.log_lr(opt, global_step)
            global_step += 1

        # ===== Validation per epoch =====
        model.eval()
        val_losses, per_class_dice_accum = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                logits_val = model(x_val)
                val_losses.append(loss_fn(cfg, logits_val, y_val).item())

                # returns a list/1D array per batch (per-class dice, typically foreground-only)
                dices = dice_per_class(
                    logits_val, y_val, cfg.num_classes,
                    ignore_background=(cfg.binary is False)
                )
                per_class_dice_accum.append(np.array(dices, dtype=np.float32))

        mean_val_loss = float(np.mean(val_losses)) if val_losses else math.nan

        # --- make sure we produce a scalar for the main metric ---
        if per_class_dice_accum:
            arr = np.stack(per_class_dice_accum, axis=0)  # shape: (num_batches, num_classes_eff)
            class_means = arr.mean(axis=0)  # per-class vector
            mean_fg_dice = float(class_means.mean())  # single scalar (overall FG mean)
        else:
            class_means = None
            mean_fg_dice = math.nan

        # --- epoch logs ---
        logger.log_scalar("val/loss", mean_val_loss, epoch)
        logger.log_scalar("val/dice_mean_fg", mean_fg_dice, epoch)  # scalar ✅
        if class_means is not None:
            for ci, d in enumerate(class_means, start=1):
                logger.log_scalar(f"val/dice_c{ci}", float(d), epoch)

        # one validation overlay (last batch) each epoch
        if 'x_val' in locals():
            logger.log_overlay(x_val, logits_val, y_val, step=epoch, tag="viz_val/img_gt_pred", max_items=4)

        print(f"[{cfg.dataset}] Epoch {epoch}/{cfg.epochs} | "
              f"val_loss={mean_val_loss:.4f} | val_dice_fg_mean={mean_fg_dice:.4f}")

        # Save best by Dice
        if mean_fg_dice > best_val_dice:
            best_val_dice = mean_fg_dice
            ckpt_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_best.pt")
            save_ckpt({"model": model.state_dict(), "epoch": epoch,
                       "val_dice": best_val_dice, "cfg": vars(cfg)}, ckpt_path)

    # Save last
    last_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_last.pt")
    save_ckpt({"model": model.state_dict(), "epoch": cfg.epochs, "cfg": vars(cfg)}, last_path)
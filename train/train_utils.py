
import os, math, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from TCSAFormer.TCSAFormer import TCSAFormer
from TrainConfig import TrainConfig
from components.SSBlock import SSBlock
from logging_utils import dice_per_class, TBLogger
from speed_banchmark import benchmark_training


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

def build_optim_sched(cfg: TrainConfig, model, steps_per_epoch):
    if cfg.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.use_cosine:
        total_steps = cfg.epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        sched = None
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

def train_one_dataset(cfg: TrainConfig, device):
    train_loader, val_loader = build_dataloaders(cfg, augment_train=True)
    model = build_model(cfg, device)
    opt, sched = build_optim_sched(cfg, model, steps_per_epoch=len(train_loader))

    benchmark_training(model, train_loader, opt, sched, loss_fn, cfg, device)

    # --- TensorBoard logger ---
    tb_dir = os.path.join(cfg.save_dir, "tb")
    logger = TBLogger(tb_dir)

    best_val_dice = -1.0
    global_step = 0
    overlay_every = 200       # steps
    hist_every = 200          # steps
    attention_every = 200     # steps

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)                # (N, C, H, W)

            if global_step % attention_every == 0:
                try:
                    from train.logging_utils import log_ssblock_attn
                    # pick a couple of SSBlocks to visualize
                    blocks_to_log = []
                    for name, m in model.named_modules():
                        if isinstance(m, SSBlock):
                            blocks_to_log.append(m)

                    for i, blk in enumerate(blocks_to_log):
                        log_ssblock_attn(logger.w, blk, global_step, tag_prefix=f"attn/block{i}",
                                         heads=min(blk.mha_cmp.h, 4))
                except Exception as e:
                    # don't crash training if logging fails
                    print(f"[warn] attention logging failed: {e}")

            loss = loss_fn(cfg, logits, y)
            loss.backward()

            # log gradient histograms occasionally
            logger.log_grads_hist(model, global_step, every_n=hist_every)

            opt.step()
            if sched is not None:
                sched.step()

            # --- step logs ---
            logger.log_scalar("train/loss", loss, global_step)
            logger.log_lr(opt, global_step)

            # overlay (image | GT | Pred) occasionally
            if global_step % overlay_every == 0:
                logger.log_overlay(x, logits, y, step=global_step, tag="viz_train/img_gt_pred", max_items=4)

            # weight histograms occasionally
            logger.log_weights_hist(model, global_step, every_n=hist_every)

            print(f"Step {step}: loss = {loss.item():.4f}")

            global_step += 1

        # ===== Validation (per epoch) =====
        model.eval()
        val_losses = []
        per_class_dice_accum = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                logits_val = model(x_val)
                val_losses.append(loss_fn(cfg, logits_val, y_val).item())

                # per-class Dice (foreground by default)
                dices = dice_per_class(logits_val, y_val, cfg.num_classes, ignore_background=cfg.binary is False)
                per_class_dice_accum.append(dices)

            mean_val_loss = float(np.mean(val_losses)) if val_losses else math.nan

            mean_per_class = None
            mean_fg_dice = math.nan
            if per_class_dice_accum:
                # average over batches → list of per-class dice
                mean_per_class = np.array(per_class_dice_accum).mean(axis=0).tolist()
                mean_fg_dice = float(np.mean(mean_per_class))

            # --- epoch logs ---
            logger.log_scalar("val/loss", mean_val_loss, epoch)
            logger.log_scalar("val/dice_mean_fg", mean_fg_dice, epoch)
            if mean_per_class is not None:
                for ci, d in enumerate(mean_per_class, start=1):   # c1.. excludes background if ignore=True
                    logger.log_scalar(f"val/dice_c{ci}", d, epoch)

            # one validation overlay (last batch) each epoch
            if 'x_val' in locals():
                logger.log_overlay(x_val, logits_val, y_val, step=epoch, tag="viz_val/img_gt_pred", max_items=4)

        print(f"[{cfg.dataset}] Epoch {epoch}/{cfg.epochs} | val_loss={mean_val_loss:.4f} | val_dice_fg_mean={mean_fg_dice:.4f}")

        # save the best checkpoint by Dice
        if mean_fg_dice > best_val_dice:
            best_val_dice = mean_fg_dice
            ckpt_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_best.pt")
            save_ckpt({"model": model.state_dict(), "epoch": epoch, "val_dice": best_val_dice, "cfg": vars(cfg)}, ckpt_path)

    # save last
    last_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_last.pt")
    save_ckpt({"model": model.state_dict(), "epoch": cfg.epochs, "cfg": vars(cfg)}, last_path)
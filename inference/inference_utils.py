# train/infer.py
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from train.TrainConfig import TrainConfig
from train.train_utils import build_model  # must exist

# ---------- Utilities ----------

def _to_tensor_rgb(resize_hw: int):
    """Minimal inference transform: resize -> tensor [0..1]."""
    return transforms.Compose([
        transforms.Resize((resize_hw, resize_hw), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

def _colorize_mask(mask_np: np.ndarray, num_classes: int) -> Image.Image:
    """
    Turn an HxW class-index mask into a color image for visualization.
    For binary, class 0=black, class 1=green; for multi-class, simple distinct hues.
    """
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if num_classes <= 2:
        out[mask_np == 1] = (0, 200, 0)        # green foreground
    else:
        palette = np.array([
            (0, 0, 0),        # 0 bg
            (220, 20, 60),    # 1
            (0, 128, 0),      # 2
            (0, 0, 128),      # 3
            (255, 165, 0),    # 4
            (128, 0, 128),    # 5
            (0, 128, 128),    # 6
            (128, 128, 0),    # 7
            (70, 130, 180),   # 8
            (255, 105, 180),  # 9
        ], dtype=np.uint8)
        for c in range(num_classes):
            color = palette[c % len(palette)]
            out[mask_np == c] = color
    return Image.fromarray(out, mode="RGB")

def _overlay(img_rgb: Image.Image, mask_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    img = np.asarray(img_rgb).astype(np.float32)
    msk = np.asarray(mask_rgb).astype(np.float32)
    over = (alpha * msk + (1.0 - alpha) * img).clip(0, 255).astype(np.uint8)
    return Image.fromarray(over)

# ---------- Model loading ----------

def load_model_for_inference(cfg: TrainConfig, ckpt_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Build the model from cfg and load weights.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # supports raw state_dict too
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print("[infer] missing keys:", missing)
    if unexpected:print("[infer] unexpected keys:", unexpected)
    model.eval()
    return model

# ---------- Single image ----------

@torch.no_grad()
def infer_image(
    cfg: TrainConfig,
    model: torch.nn.Module,
    image_path: str,
    out_mask_path: Optional[str] = None,
    out_overlay_path: Optional[str] = None,
    threshold: float = 0.5,  # used for binary
    keep_size: bool = True,  # resize mask back to original size
) -> np.ndarray:
    """
    Run inference on a single RGB image file.
    Returns the mask as numpy (H, W) with class indices.
    """
    device = next(model.parameters()).device
    img_pil = Image.open(image_path).convert("RGB")
    W0, H0 = img_pil.size

    tf = _to_tensor_rgb(cfg.resolution)
    x = tf(img_pil).unsqueeze(0).to(device, non_blocking=True, memory_format=torch.channels_last)

    # mixed precision is fine for inference
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype) if device != "cpu" else torch.no_grad():
        logits = model(x)                          # (1, C, H, W)
        if cfg.binary or cfg.num_classes == 2:
            # binary: probs = sigmoid on channel 1 or channel 0? Assume model outputs C=2 (softmax later)
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # foreground prob (1,1,H,W)
            mask_small = (probs >= threshold).long().squeeze(0).squeeze(0)  # (H,W)
        else:
            # multi-class
            mask_small = torch.argmax(logits, dim=1).squeeze(0)  # (H,W)

    # resize mask back to original image size (nearest)
    if keep_size:
        mask = F.interpolate(
            mask_small.unsqueeze(0).unsqueeze(0).float(),
            size=(H0, W0),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()
    else:
        mask = mask_small

    mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    # save outputs if requested
    if out_mask_path:
        Image.fromarray(mask_np, mode="L").save(out_mask_path)

    if out_overlay_path:
        mask_rgb = _colorize_mask(mask_np, cfg.num_classes)
        if keep_size:
            mask_rgb = mask_rgb.resize((W0, H0), resample=Image.NEAREST)
        over = _overlay(img_pil, mask_rgb, alpha=0.45)
        over.save(out_overlay_path)

    return mask_np

# ---------- Folder inference ----------

@torch.no_grad()
def infer_folder(
    cfg: TrainConfig,
    ckpt_path: str,
    in_dir: str,
    out_dir_masks: str,
    out_dir_overlays: Optional[str] = None,
    exts: Optional[List[str]] = None,
    threshold: float = 0.5,
    keep_size: bool = True,
    device: Optional[str] = None,
):
    """
    Run inference on all images in a folder.
    Saves masks as PNG (L mode with class indices).
    Optionally saves overlays.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(cfg, ckpt_path, device=device)

    in_dir = Path(in_dir)
    out_dir_masks = Path(out_dir_masks)
    out_dir_masks.mkdir(parents=True, exist_ok=True)
    if out_dir_overlays:
        out_dir_overlays = Path(out_dir_overlays)
        out_dir_overlays.mkdir(parents=True, exist_ok=True)

    exts = exts or [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    files = [p for p in in_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files.sort()

    for p in files:
        stem = p.stem
        out_mask_path = str(out_dir_masks / f"{stem}.png")
        out_overlay_path = str(out_dir_overlays / f"{stem}.png") if out_dir_overlays else None

        infer_image(
            cfg=cfg,
            model=model,
            image_path=str(p),
            out_mask_path=out_mask_path,
            out_overlay_path=out_overlay_path,
            threshold=threshold,
            keep_size=keep_size,
        )
        print(f"[infer] {p.name} -> {out_mask_path}" + (f", {out_overlay_path}" if out_overlay_path else ""))


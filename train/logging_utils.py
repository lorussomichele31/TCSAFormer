from pathlib import Path
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value, step: int):
        self.w.add_scalar(tag, value.item() if torch.is_tensor(value) else float(value), step)

    def log_lr(self, opt, step: int, tag="train/lr"):
        self.w.add_scalar(tag, opt.param_groups[0]["lr"], step)

    def log_weights_hist(self, model, step: int, every_n: int = 200):
        if step % every_n: return
        for n, p in model.named_parameters():
            if p.requires_grad and p.numel():
                self.w.add_histogram(f"weights/{n}", p.detach().cpu(), step)

    def log_grads_hist(self, model, step: int, every_n: int = 200):
        if step % every_n: return
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.w.add_histogram(f"grads/{n}", p.grad.detach().cpu(), step)

    @torch.no_grad()
    def log_overlay(self, images, logits, masks, step: int, tag="viz/img_gt_pred", max_items: int = 4):
        """
        images: [B,3,H,W] in [0,1]; logits: [B,C,H,W]; masks: [B,H,W]
        Creates a grid: per sample -> image | GT | Pred
        """
        B = min(images.size(0), max_items)
        imgs = images[:B].detach().cpu().clamp(0, 1)
        preds = logits[:B].detach().softmax(1).argmax(1).cpu()  # [B,H,W]
        gts   = masks[:B].detach().cpu()                        # [B,H,W]

        def to_color(mask):
            # tiny palette
            palette = torch.tensor([
                [0,0,0], [220,20,60], [0,128,0], [0,0,128], [255,165,0],
                [255,20,147],[70,130,180],[128,0,128],[255,0,0],[0,255,0]
            ], dtype=torch.float32)  # extend if many classes
            mask = mask.clamp(max=palette.size(0)-1)
            col = palette[mask.long()]
            return (col/255.).permute(2,0,1)  # [3,H,W]

        gt_col   = torch.stack([to_color(m) for m in gts], 0)
        pred_col = torch.stack([to_color(p) for p in preds], 0)

        panel = torch.cat([imgs, gt_col, pred_col], dim=0)  # [3B,3,H,W]
        grid = torchvision.utils.make_grid(panel, nrow=B, pad_value=1.0)
        self.w.add_image(tag, grid, step)


@torch.no_grad()
def dice_per_class(logits, target, num_classes: int, ignore_background: bool = True):
    """Returns a list of per-class Dice (float)."""
    pred = logits.argmax(1)  # [B,H,W]
    dices = []
    start_c = 1 if ignore_background else 0
    for c in range(start_c, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float() * 2
        denom = p.sum().float() + t.sum().float() + 1e-8
        dices.append((inter/denom).item())
    return dices


@torch.no_grad()
def log_ssblock_attn(
    writer,
    block,
    step: int,
    tag_prefix: str = "attn",
    heads: int = 4,
    query: str | tuple = "center",  # "center" or (qy, qx)
):
    """
    Logs 3 panels for a single SSBlock:
      1) Local window attention (from SSBlock):   [heads, win, win]
      2) Global compressed attention (mha_cmp):   [heads, nH, nW]
      3) Selected blocks attention (mha_sel):     [heads, nH, nW] (zeros except top-k)

    Expects the SSBlock / SSMHA to expose:
      block.last_H, block.last_W, block.last_nH, block.last_nW, block.last_sel_idx
      block.last_attn_win:            [B, Nq, H_heads, win*win]
      block.mha_cmp.last_attn:        [B, H_heads, Nq, Nc (nH*nW)]
      block.mha_sel.last_attn:        [B, H_heads, Nq, K (topk)]
    """
    # --- Basic availability checks ---
    H = getattr(block, "last_H", None)
    W = getattr(block, "last_W", None)
    nH = getattr(block, "last_nH", None)
    nW = getattr(block, "last_nW", None)

    if H is None or W is None or nH is None or nW is None:
        return  # not run yet

    # Pick query pixel (y, x)
    if query == "center":
        qy, qx = H // 2, W // 2
    else:
        qy, qx = query
        qy = int(max(0, min(H - 1, qy)))
        qx = int(max(0, min(W - 1, qx)))

    q_idx = qy * W + qx  # flatten index in Nq=H*W

    # Helper: normalize a [N, H, W] tensor per-item to [0,1]
    def _norm_3d(t):
        t = t - t.amin(dim=(-1, -2), keepdim=True)
        denom = t.amax(dim=(-1, -2), keepdim=True) + 1e-8
        return t / denom

    # =========================
    # 1) Local window attention
    # =========================
    if getattr(block, "last_attn_win", None) is not None:
        a_win = block.last_attn_win  # [B, Nq, H_heads, win*win] (on CPU)
        if a_win.ndim == 4 and a_win.size(1) >= q_idx + 1:
            H_heads = a_win.size(2)
            use_heads = min(heads, H_heads)
            win = int((a_win.size(-1)) ** 0.5)
            a = a_win[0, q_idx, :use_heads]         # [heads, win*win]
            a = a.view(use_heads, win, win)         # [heads, win, win]
            a = _norm_3d(a)
            grid_win = torchvision.utils.make_grid(a.unsqueeze(1), nrow=use_heads, pad_value=1.0)
            writer.add_image(f"{tag_prefix}/local_window", grid_win, step)

    # =======================================
    # 2) Global compressed attention (mha_cmp)
    # =======================================
    a_cmp_all = getattr(block.mha_cmp, "last_attn", None)
    if a_cmp_all is not None:
        # expected shape: [B, H_heads, Nq, Nc] with Nc = nH*nW
        if a_cmp_all.ndim == 4 and a_cmp_all.size(2) >= q_idx + 1 and a_cmp_all.size(3) == nH * nW:
            H_heads = a_cmp_all.size(1)
            use_heads = min(heads, H_heads)
            a_cmp = a_cmp_all[0, :use_heads, q_idx]  # [heads, Nc]
            a_cmp = a_cmp.view(use_heads, nH, nW)    # [heads, nH, nW]
            a_cmp = _norm_3d(a_cmp)
            grid_cmp = torchvision.utils.make_grid(a_cmp.unsqueeze(1), nrow=use_heads, pad_value=1.0)
            writer.add_image(f"{tag_prefix}/global_blocks", grid_cmp, step)

    # ========================================
    # 3) Selected blocks attention (mha_sel)
    #    projected back onto the nH×nW grid
    # ========================================
    a_sel_all = getattr(block.mha_sel, "last_attn", None)
    sel_idx = getattr(block, "last_sel_idx", None)
    if a_sel_all is not None and sel_idx is not None:
        # a_sel_all: [B, H_heads, Nq, K], sel_idx: [B, K] with values in [0, nH*nW)
        if (a_sel_all.ndim == 4 and a_sel_all.size(2) >= q_idx + 1
                and sel_idx.ndim == 2 and sel_idx.size(0) >= 1):
            H_heads = a_sel_all.size(1)
            use_heads = min(heads, H_heads)
            K = a_sel_all.size(3)
            a_sel = a_sel_all[0, :use_heads, q_idx]  # [heads, K]
            idx = sel_idx[0]                         # [K]
            # place onto full grid (nH*nW), zeros elsewhere
            full = torch.zeros((use_heads, nH * nW), dtype=a_sel.dtype)
            full[:, idx] = a_sel
            full = full.view(use_heads, nH, nW)
            full = _norm_3d(full)
            grid_sel = torchvision.utils.make_grid(full.unsqueeze(1), nrow=use_heads, pad_value=1.0)
            writer.add_image(f"{tag_prefix}/selected_blocks", grid_sel, step)

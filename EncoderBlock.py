# SPDX-License-Identifier: MIT
# SpatialSparseEncoder2D: a 2D adaptation of Spatial Sparse Attention ideas
# for efficient medical image segmentation encoders.
#
# Key pieces:
# - Overlapping patch embedding (conv stem -> NHWC + LayerNorm)
# - Block partitioning -> compressed tokens (global-ish K/V)
# - Selection path: top-k blocks per image (shared for all queries) -> K/V
# - Local path: per-query local window attention (per-query K/V)
# - Per-query gating to fuse (compressed, selection, local)
# - Optional KL on compressed tokens (VAE-ish), off by default

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# -------------------- Small utilities --------------------

class OverlapPatchEmbed(nn.Module):
    """Overlap patch embedding (conv stem).
    Input:  (B, C_in, H, W)
    Output: (B, H', W', C)  (NHWC with LayerNorm on C)
    """
    def __init__(self, in_ch: int, embed_dim: int, kernel_size: int = 7, stride: int = 4, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size, stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
        x = self.norm(x)
        return x, H, W


def _hw_to_blocks(H: int, W: int, block: int) -> Tuple[int, int]:
    assert H % block == 0 and W % block == 0, f"H,W must be multiples of block (got {H},{W} vs {block})"
    return H // block, W // block


# -------------------- Attention primitives --------------------

class MHA(nn.Module):
    """Multi-head attention over (Q,K,V) in shape:
         Q: (B, Nq, C),  K: (B, Nk, C),  V: (B, Nk, C)  ->  (B, Nq, C)
    """
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.drop_proj = nn.Dropout(proj_drop)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Nq, C = Q.shape
        Nk = K.shape[1]
        h, d = self.h, self.d

        q = self.Wq(Q).view(B, Nq, h, d).transpose(1, 2)    # (B,h,Nq,d)
        k = self.Wk(K).view(B, Nk, h, d).transpose(1, 2)    # (B,h,Nk,d)
        v = self.Wv(V).view(B, Nk, h, d).transpose(1, 2)    # (B,h,Nk,d)

        attn = (q @ k.transpose(-1, -2)) * self.scale       # (B,h,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.drop_attn(attn)
        out = attn @ v                                      # (B,h,Nq,d)
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)
        out = self.proj(out)
        out = self.drop_proj(out)
        return out


class Gating(nn.Module):
    """Per-query gate over 3 sources (compressed, selected, local)."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, q_feat: torch.Tensor, y_cmp: torch.Tensor, y_sel: torch.Tensor, y_win: torch.Tensor) -> torch.Tensor:
        # q_feat, y_*: (B,Nq,C)
        w = self.fc(q_feat).softmax(dim=-1)                 # (B,Nq,3)
        y = w[..., 0:1] * y_cmp + w[..., 1:2] * y_sel + w[..., 2:3] * y_win
        return y


# -------------------- Spatial Sparse Encoder Block (2D) --------------------

@dataclass
class SS2DConfig:
    embed_dim: int = 96
    heads: int = 3
    block_size: int = 8          # block size on H,W after embedding
    topk_blocks: int = 4         # top-k blocks per image for selection path
    win_size: int = 7            # local window (odd)
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_kl: bool = False         # optional KL on compressed tokens
    kl_beta: float = 1e-4


class SpatialSparseEncoder2D(nn.Module):
    """
    Input:  (B, C_in, H, W)
    Output: dict:
       'feat'  : (B, H', W', C)  fused token features (NHWC)
       'latent': (B, nBlocks, C) compressed block tokens
       'kl'    : scalar tensor (0 if use_kl=False)
       'meta'  : dict with shapes
    """
    def __init__(self, in_ch: int, cfg: SS2DConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim

        # 1) Overlap embedding
        self.patch = OverlapPatchEmbed(in_ch, C, kernel_size=7, stride=4)

        # 2) Compressed tokens (VAE-ish heads)
        self.comp_mu   = nn.Linear(C, C)
        self.comp_logv = nn.Linear(C, C)

        # 3) Attentions
        self.mha_cmp = MHA(C, cfg.heads, cfg.attn_drop, cfg.proj_drop)  # compressed/global
        self.mha_sel = MHA(C, cfg.heads, cfg.attn_drop, cfg.proj_drop)  # selection blocks
        self.mha_win = MHA(C, cfg.heads, cfg.attn_drop, cfg.proj_drop)  # used only for projections in local path

        # 4) Fusion + FFN
        self.gate = Gating(C)
        self.ffn  = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.GELU(),
            nn.Linear(4*C, C),
        )
        self.norm_q = nn.LayerNorm(C)
        self.norm_o = nn.LayerNorm(C)

        # 5) Scorer for blocks (selection path)
        self.block_scorer = nn.Sequential(
            nn.Linear(C, C//2),
            nn.GELU(),
            nn.Linear(C//2, 1)
        )

    # ----- internals -----

    def _make_blocks(self, feat: torch.Tensor, H: int, W: int, block: int):
        """Group NHWC feat into blocks.
        Returns:
          x (B, nH, nW, block, block, C),
          x_flat (B, nB, block*block, C),
          blk_mean (B, nB, C),
          nH, nW
        """
        B, _, _, C = feat.shape
        nH, nW = _hw_to_blocks(H, W, block)
        x = feat.view(B, nH, block, nW, block, C).permute(0, 1, 3, 2, 4, 5)  # (B, nH, nW, b, b, C)
        x_flat = x.reshape(B, nH*nW, block*block, C)                          # (B, nB, B^2, C)
        blk_mean = x_flat.mean(dim=2)                                         # (B, nB, C)
        return x, x_flat, blk_mean, nH, nW

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma^2) || N(0,1)), averaged
        return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    def _gather_local_window(self, feat_flat: torch.Tensor, H: int, W: int, win: int):
        """For each query token, gather local win×win neighborhood as K,V.
        Returns: (K_win, V_win) each (B, Nq, win^2, C)
        """
        B, N, C = feat_flat.shape
        pad = win // 2
        # NHWC -> NCHW for unfold
        feat = feat_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        feat = F.pad(feat, (pad, pad, pad, pad), mode='replicate')  # pad W then H

        patches = F.unfold(feat, kernel_size=win, stride=1)  # (B, C*win*win, H*W)
        patches = patches.transpose(1, 2).contiguous()       # (B, Nq, C*win^2)
        patches = patches.view(B, N, C, win*win).permute(0, 1, 3, 2)  # (B, Nq, win^2, C)
        return patches, patches  # K, V

    def _local_attend(self, Q: torch.Tensor, K_win: torch.Tensor, V_win: torch.Tensor) -> torch.Tensor:
        """
        Per-query local attention (each query attends only to its own win^2 keys).
        Q:     (B, Nq, C)
        K_win: (B, Nq, W2, C)
        V_win: (B, Nq, W2, C)
        Return: (B, Nq, C)
        """
        B, Nq, C = Q.shape
        W2 = K_win.shape[2]
        h = self.mha_win.h
        d = self.mha_win.d
        scale = d ** -0.5

        # Project
        q = self.mha_win.Wq(Q).view(B, Nq, h, d)                    # (B,Nq,h,d)
        k = self.mha_win.Wk(K_win).view(B, Nq, W2, h, d)            # (B,Nq,W2,h,d)
        v = self.mha_win.Wv(V_win).view(B, Nq, W2, h, d)            # (B,Nq,W2,h,d)

        # Merge batch & query so each query has its own small attention
        q = q.permute(0, 1, 2, 3).contiguous().view(B * Nq, h, 1,  d)    # (B*Nq,h,1,d)
        k = k.permute(0, 1, 3, 2, 4).contiguous().view(B * Nq, h, W2, d) # (B*Nq,h,W2,d)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(B * Nq, h, W2, d) # (B*Nq,h,W2,d)

        attn = (q @ k.transpose(-1, -2)) * scale                        # (B*Nq,h,1,W2)
        attn = attn.softmax(dim=-1)
        attn = self.mha_win.drop_attn(attn)
        out  = attn @ v                                                 # (B*Nq,h,1,d)
        out  = out.squeeze(-2).transpose(1, 2).contiguous().view(B * Nq, C)  # (B*Nq,C)
        out  = self.mha_win.proj(out)
        out  = self.mha_win.drop_proj(out).view(B, Nq, C)                    # (B,Nq,C)
        return out

    # ----- forward -----

    def forward(self, x: torch.Tensor):
        """
        x: (B, C_in, H, W)
        """
        cfg = self.cfg
        win = cfg.win_size
        assert win % 2 == 1, "win_size must be odd"

        # 1) embed
        feat, H, W = self.patch(x)                  # (B,H,W,C) NHWC
        B, _, _, C = feat.shape

        # 2) ensure H,W divisible by block
        block = cfg.block_size
        pad_h = (block - (H % block)) % block
        pad_w = (block - (W % block)) % block
        if pad_h or pad_w:
            # NHWC -> NCHW to pad spatial dims correctly
            feat_nchw = feat.permute(0, 3, 1, 2)     # (B,C,H,W)
            feat_nchw = F.pad(feat_nchw, (0, pad_w, 0, pad_h))  # pad W then H
            feat = feat_nchw.permute(0, 2, 3, 1)     # back to NHWC
            H += pad_h
            W += pad_w

        Nq = H * W
        feat_flat = feat.view(B, Nq, C)

        # 3) blocks & compressed (global-ish) tokens
        _, blk_flat, blk_mean, nH, nW = self._make_blocks(feat, H, W, block)  # blk_mean: (B,nB,C)
        nB = blk_mean.shape[1]

        mu   = self.comp_mu(blk_mean)               # (B,nB,C)
        logv = self.comp_logv(blk_mean)             # (B,nB,C)

        if cfg.use_kl:
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            z   = mu + eps * std
            kl  = self._kl(mu, logv) * cfg.kl_beta
        else:
            z  = mu
            kl = torch.tensor(0.0, device=x.device)

        # 4) selection path: score blocks -> top-k per image (shared across all queries)
        scores = self.block_scorer(blk_mean).squeeze(-1)      # (B,nB)
        topk = min(cfg.topk_blocks, nB)
        sel_vals, sel_idx = scores.topk(topk, dim=1)          # (B,topk)

        K_sel = torch.gather(z, 1, sel_idx.unsqueeze(-1).expand(B, topk, C))  # (B,topk,C)
        V_sel = K_sel

        # 5) compressed/global path
        K_cmp, V_cmp = z, z   # (B,nB,C)

        # 6) local window path (per-query)
        K_win, V_win = self._gather_local_window(feat_flat, H, W, win)        # (B,Nq,win^2,C)

        # 7) attention on three sources
        Q = feat_flat                                                         # (B,Nq,C)
        y_cmp = self.mha_cmp(Q, K_cmp, V_cmp)                                 # (B,Nq,C)
        y_sel = self.mha_sel(Q, K_sel, V_sel)                                 # (B,Nq,C)
        y_win = self._local_attend(Q, K_win, V_win)                           # (B,Nq,C)

        # 8) fuse + FFN
        qn = self.norm_q(Q)
        y  = self.gate(qn, y_cmp, y_sel, y_win)
        y  = y + Q
        y  = self.norm_o(y + self.ffn(y))

        # 9) reshape back to NHWC
        feat_out = y.view(B, H, W, C)

        # 10) crop padding (if any)
        if pad_h or pad_w:
            feat_out = feat_out[:, :H - pad_h, :W - pad_w, :]
            H -= pad_h
            W -= pad_w

        return {
            "feat": feat_out,                     # (B, H', W', C)
            "latent": z,                          # (B, nBlocks, C)
            "kl": kl,                             # scalar tensor
            "meta": {"H": H, "W": W, "block": block, "nH": nH, "nW": nW}
        }


# -------------------- minimal smoke test --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SS2DConfig(
        embed_dim=96,
        heads=3,
        block_size=8,
        topk_blocks=4,
        win_size=7,
        attn_drop=0.0,
        proj_drop=0.0,
        use_kl=True
    )

    enc = SpatialSparseEncoder2D(in_ch=1, cfg=cfg)
    x = torch.randn(2, 1, 256, 256)  # (B,C,H,W)
    out = enc(x)

    feat = out["feat"]        # (B,H',W',C)  -> for decoders, usually permute to (B,C,H',W')
    latent = out["latent"]    # (B,nBlocks,C)
    kl = out["kl"]

    summary(enc)

    print("feat:", feat.shape)
    print("latent:", latent.shape)
    print("kl:", float(kl))
    # Example: quick decoder head
    feat_nchw = feat.permute(0, 3, 1, 2)               # (B,C,H',W') = (2,96,64,64) for stride=4 stem
    head = nn.Sequential(
        nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GELU(),   # 64x128x128
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),   # 32x256x256
        nn.Conv2d(32, 1, 1)
    )
    logits = head(feat_nchw)
    print("logits:", logits.shape)  # (2,1,256,256)

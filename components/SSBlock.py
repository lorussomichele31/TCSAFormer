
from components.Attention import SSGating, SSMHA
from components.DBFFN import DBFFN
from timm.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSBlock(nn.Module):
    """
    Spatial-Sparse block (encoder/decoder compatible):
      - Input/Output: NCHW (B,C,H,W)
      - Internally: NHWC for LayerNorm + DBFFN
      - 3 attention paths: compressed global (all blocks), selected blocks (top-k),
        local window (per-query).
      - Exposes tensors for logging:
          * self.last_H, self.last_W
          * self.last_nH, self.last_nW
          * self.last_sel_idx           # [B, topk]
          * self.last_attn_win          # [B, H*W, heads, win*win]
        (Plus, SSMHA.mha_cmp/mha_sel must cache .last_attn: [B, heads, Nq, Nk])
    """
    def __init__(self,
                 dim: int,
                 drop_path: float = 0.0,
                 num_heads: int = 8,
                 before_attn_dwconv: int = 3,
                 block_size: int = 8,
                 topk_blocks: int = 4,
                 win_size: int = 7,
                 ):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.topk = topk_blocks
        self.win = win_size
        assert self.win % 2 == 1, "win_size must be odd"

        # lightweight depthwise pos-encoding
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.mha_cmp = SSMHA(dim, num_heads)  # compressed/global (all blocks)
        self.mha_sel = SSMHA(dim, num_heads)  # selected top-k blocks
        self.mha_win = SSMHA(dim, num_heads)  # projections for local path

        self.gate = SSGating(dim)
        self.mlp = DBFFN(dim=dim, bias=False, ffn_expansion_factor=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.block_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

        # ---------- Exposed for logging ----------
        self.last_H = None
        self.last_W = None
        self.last_nH = None
        self.last_nW = None
        self.last_sel_idx = None           # [B, topk]
        self.last_attn_win = None          # [B, H*W, heads, win*win]

    @staticmethod
    def _to_nhwc(x): return x.permute(0, 2, 3, 1)
    @staticmethod
    def _to_nchw(x): return x.permute(0, 3, 1, 2)

    def _blocks_mean(self, feat_nhwc, H, W):
        B, H_, W_, C = feat_nhwc.shape
        assert H_ == H and W_ == W
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0, "H,W must be multiples of block_size"
        nH, nW = H // bs, W // bs
        x = feat_nhwc.view(B, nH, bs, nW, bs, C).permute(0, 1, 3, 2, 4, 5)  # (B,nH,nW,bs,bs,C)
        x = x.reshape(B, nH * nW, bs * bs, C)                                # (B,nB,B^2,C)
        blk_mean = x.mean(dim=2)                                             # (B,nB,C)

        # cache grid size for logging
        self.last_nH, self.last_nW = nH, nW
        return blk_mean, nH, nW

    def _gather_local(self, feat_flat, H, W):
        B, Nq, C = feat_flat.shape
        pad = self.win // 2
        f = feat_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        f = F.pad(f, (pad, pad, pad, pad), mode='replicate')
        patches = F.unfold(f, kernel_size=self.win, stride=1)                # (B, C*win^2, H*W)
        patches = patches.transpose(1, 2).contiguous()                       # (B, Nq, C*win^2)
        patches = patches.view(B, Nq, C, self.win * self.win).permute(0, 1, 3, 2)  # (B,Nq,win^2,C)
        return patches

    def _local_attend(self, Q, K_win, V_win):
        B, Nq, C = Q.shape
        W2 = K_win.shape[2]                    # win^2
        h = self.mha_win.h
        d = self.mha_win.d
        scale = d ** -0.5

        q = self.mha_win.wq(Q).view(B, Nq, h, d)                     # (B,Nq,h,d)
        k = self.mha_win.wk(K_win).view(B, Nq, W2, h, d)             # (B,Nq,W2,h,d)
        v = self.mha_win.wv(V_win).view(B, Nq, W2, h, d)

        q = q.view(B * Nq, h, 1, d)                                  # (B*Nq,h,1,d)
        k = k.permute(0, 1, 3, 2, 4).contiguous().view(B * Nq, h, W2, d)  # (B*Nq,h,W2,d)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(B * Nq, h, W2, d)

        attn = (q @ k.transpose(-1, -2)) * scale                     # (B*Nq,h,1,W2)
        attn = attn.softmax(dim=-1)

        # cache local attention for logging: [B, Nq, h, W2]
        self.last_attn_win = attn.view(B, Nq, h, W2).detach().cpu()

        out = (attn @ v).squeeze(-2).transpose(1, 2).contiguous().view(B * Nq, h * d)
        out = self.mha_win.drop_proj(self.mha_win.proj(out)).view(B, Nq, C)
        return out

    def forward(self, x):  # NCHW
        B, C, H, W = x.shape

        # cache spatial size for logging
        self.last_H, self.last_W = H, W

        # DW positional term
        x = x + (self.pos_embed(x) if isinstance(self.pos_embed, nn.Conv2d) else 0)

        feat = self._to_nhwc(x)           # (B,H,W,C)
        Q = feat.view(B, H * W, C)        # (B,Nq,C)

        # compressed tokens (all blocks)
        blk_mean, nH, nW = self._blocks_mean(feat, H, W)  # (B,nB,C)
        nB = blk_mean.shape[1]

        # selection (top-k blocks)
        scores = self.block_scorer(blk_mean).squeeze(-1)  # (B,nB)
        topk = min(self.topk, nB)
        _, sel_idx = scores.topk(topk, dim=1)             # (B,topk)
        self.last_sel_idx = sel_idx.detach().cpu()        # cache indices for logging
        K_sel = torch.gather(blk_mean, 1, sel_idx.unsqueeze(-1).expand(B, topk, C))  # (B,topk,C)

        # three attention paths
        y_cmp = self.mha_cmp(Q, blk_mean, blk_mean)       # (B,Nq,C)  -- mha_cmp.last_attn cached in SSMHA
        y_sel = self.mha_sel(Q, K_sel, K_sel)             # (B,Nq,C)  -- mha_sel.last_attn cached in SSMHA
        K_win = self._gather_local(Q, H, W)               # (B,Nq,win^2,C)
        y_win = self._local_attend(Q, K_win, K_win)       # (B,Nq,C)  -- self.last_attn_win cached

        # fuse
        qn = self.norm1(Q)
        y = self.gate(qn, y_cmp, y_sel, y_win)            # (B,Nq,C)
        y = y + Q                                         # (B,Nq,C)

        y_map = y.view(B, H, W, C)                        # (B,H,W,C)
        y_map = self.norm2(y_map + self.mlp(y_map))       # (B,H,W,C)

        # back to NCHW
        y = self._to_nchw(y_map)                          # (B,C,H,W)
        y = self.drop_path(y)
        return y

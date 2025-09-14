import torch.nn as nn

class SSMHA(nn.Module):
    """Multi-head attention over Q:(B,Nq,C), K/V:(B,Nk,C) -> (B,Nq,C)"""
    def __init__(self, dim: int, heads: int, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.last_attn = None
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.drop_proj = nn.Dropout(proj_drop)

    def forward(self, Q, K, V):
        B, Nq, C = Q.shape
        Nk = K.shape[1]
        h, d = self.h, self.d

        q = self.wq(Q).view(B, Nq, h, d).transpose(1, 2)     # (B,h,Nq,d)
        k = self.wk(K).view(B, Nk, h, d).transpose(1, 2)     # (B,h,Nk,d)
        v = self.wv(V).view(B, Nk, h, d).transpose(1, 2)     # (B,h,Nk,d)

        attn = (q @ k.transpose(-1, -2)) * self.scale        # (B,h,Nq,Nk)
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach().cpu()
        attn = self.drop_attn(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, h * d)
        out = self.drop_proj(self.proj(out))                 # (B,Nq,C)
        return out


class SSGating(nn.Module):
    """Per-query soft gate over 3 sources (compressed, selected, local)."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, qn, y_cmp, y_sel, y_win):   # all (B,Nq,C)
        w = self.fc(qn).softmax(dim=-1)          # (B,Nq,3)
        return w[...,0:1]*y_cmp + w[...,1:2]*y_sel + w[...,2:3]*y_win


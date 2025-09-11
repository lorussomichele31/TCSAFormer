import torch.nn as nn
import torch

class DBFFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.norm_in = nn.BatchNorm2d(dim)
        self.norm_proj = nn.BatchNorm2d(hidden_features)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, padding=3,
                                   groups=hidden_features, bias=bias)
        self.act3 = nn.GELU()
        self.act7 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.dwconv3x3_1 = nn.Conv2d(hidden_features, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features, dim, kernel_size=7, padding=3, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.norm_out = nn.BatchNorm2d(dim)

    def forward(self, x):  # expects NHWC
        x = x.permute(0, 3, 1, 2)        # NHWC -> NCHW
        x = self.norm_in(x)
        x = self.project_in(x)
        x = self.norm_proj(x)

        x3_1, x3_2 = self.act3(self.dwconv3x3(x)).chunk(2, dim=1)
        x7_1, x7_2 = self.act7(self.dwconv7x7(x)).chunk(2, dim=1)

        x1 = torch.cat([x3_1, x7_1], dim=1)
        x2 = torch.cat([x3_2, x7_2], dim=1)

        x1 = self.norm1(self.dwconv3x3_1(x1))
        x2 = self.norm2(self.dwconv7x7_1(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)
        x = self.norm_out(x)
        x = x.permute(0, 2, 3, 1)        # back to NHWC
        return x


class Skip(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(Skip, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):  # NCHW
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out
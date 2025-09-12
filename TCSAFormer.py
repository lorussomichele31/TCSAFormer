
import torch.nn as nn
import torch
import torchinfo
from timm.layers import trunc_normal_
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
import torch.nn.functional as F
from components.DBFFN import Skip
from components.SSBlock import SSBlock

class TCSA_Former(nn.Module):
    def __init__(self,
                 depth=[3, 4, 8, 3],
                 in_chans=3,
                 num_classes=100,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 drop_path_rate=0.,
                 use_checkpoint_stages=[],
                 qk_dims=[None, None, None, None],
                 before_attn_dwconv=3,
                 ss_block_size=8,
                 ss_topk=4,
                 ss_win=7):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency

        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
        )

        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        nheads = [dim // head_dim for dim in qk_dims]

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim[i + 1]),
            )

            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)

        for i in range(3):
            upsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[3 - i], embed_dim[2 - i], kernel_size=1),
                nn.BatchNorm2d(embed_dim[2 - i]),
            )
            self.upsample_layers.append(upsample_layer)

        for i in range(3):
            self.skip_layers.append(Skip(embed_dim[2 - i], embed_dim[2 - i]))

        self.stages = nn.ModuleList()
        self.stages_de = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            enc_stage = nn.Sequential(
                *[SSBlock(
                    dim=embed_dim[i],
                    drop_path=dp_rates[cur + j],
                    num_heads=nheads[i],
                    before_attn_dwconv=before_attn_dwconv,
                    block_size=ss_block_size,
                    topk_blocks=ss_topk,
                    win_size=ss_win
                ) for j in range(depth[i])]
            )
            if i in use_checkpoint_stages:
                enc_stage = checkpoint_wrapper(enc_stage)
            self.stages.append(enc_stage)

            if i < 3:
                dec_stage = nn.Sequential(
                    *[SSBlock(
                        dim=embed_dim[2 - i],
                        drop_path=dp_rates[cur + j],
                        num_heads=nheads[2 - i],
                        before_attn_dwconv=before_attn_dwconv,
                        block_size=ss_block_size,
                        topk_blocks=ss_topk,
                        win_size=ss_win
                    ) for j in range(depth[2 - i])]
                )
                self.stages_de.append(dec_stage)

            cur += depth[i]

        self.output = nn.Conv2d(in_channels=embed_dim[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(x)
        for i in range(3):
            x = F.interpolate(x, size=(x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
            x = self.upsample_layers[i](x)
            x = out[2 - i] + x
            x = self.skip_layers[i](x)
            x = self.stages_de[i](x)
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        x = self.output(x)
        return x


class TCSAFormer(nn.Module):
    def __init__(self, num_classes=1):
        super(TCSAFormer, self).__init__()
        self.TCSAFormer = TCSA_Former(
            depth=[2, 2, 8, 2],
            embed_dim=[64, 128, 256, 512],
            before_attn_dwconv=3,
            num_classes=num_classes,
            qk_dims=[64, 128, 256, 512],
            head_dim=32,
            use_checkpoint_stages=[],
            drop_path_rate=0.2,
            # SSBlock knobs (tune freely)
            ss_block_size=8,
            ss_topk=4,
            ss_win=7,
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.TCSAFormer(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TCSAFormer(
        num_classes=1,
    ).to(device)

    x = torch.randn((1, 3, 256, 256)).to(device)
    out = model(x)
    print(out.shape)
    torchinfo.summary(model)
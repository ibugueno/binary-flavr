import torch
import torch.nn as nn
from .binconv import BinConv3d

class SEGating(nn.Module):
    def __init__(self , inplanes , reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            BinConv3d(in_ch=inplanes, out_ch=inplanes,
                      kernel_size=1, stride=1, padding=0,
                      bias=True, batchnorm=False),
            nn.Sigmoid()
        )

    def forward(self , x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y

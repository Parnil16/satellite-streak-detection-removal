"""
model.py
U-Net architecture — must match the training definition exactly.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool(2) → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x  = self.up(x)
        # Pad to match skip's spatial size (handles odd dimensions)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = nn.functional.pad(x, [dw // 2, dw - dw // 2,
                                    dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    in_channels  : 1  (grayscale)
    out_channels : 1  (streak logit — apply sigmoid externally)
    features     : encoder channel widths at each level
    """

    def __init__(
        self,
        in_channels:  int  = 1,
        out_channels: int  = 1,
        features: tuple    = (32, 64, 128, 256),
    ):
        super().__init__()

        self.enc1       = DoubleConv(in_channels, features[0])
        self.enc2       = Down(features[0], features[1])
        self.enc3       = Down(features[1], features[2])
        self.enc4       = Down(features[2], features[3])
        self.bottleneck = Down(features[3], features[3] * 2)

        self.dec4 = Up(features[3] * 2 + features[3], features[3])
        self.dec3 = Up(features[3]     + features[2], features[2])
        self.dec2 = Up(features[2]     + features[1], features[1])
        self.dec1 = Up(features[1]     + features[0], features[0])

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)
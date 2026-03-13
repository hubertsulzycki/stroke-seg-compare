import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    A building block for U-Net consisting of two consecutive Convolutional layers,
    each followed by Batch Normalization and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_layers(x)


class Up(nn.Module):
    """
    Upscaling followed by DoubleConv.
    Applies ConvTranspose2d for spatial upsampling, concatenates with the
    corresponding skip connection from the encoder, and smooths the features
    using a DoubleConv block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 is the tensor from the decoder (below)
        # x2 is the skip connection tensor from the encoder (side)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        # Encoder part of 2D UNet model.
        # Slices transformations: (1,512,512) -> (64,512,512) -> (128, 256, 256) -> (256, 128, 128) -> (512, 64, 64)
        self.inc = DoubleConv(in_channels=in_channels, out_channels=64)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=64, out_channels=128),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=128, out_channels=256),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=256, out_channels=512),
        )

        # Bottleneck part of 2D UNet model
        # Final transformation to (1024, 32, 32)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=512, out_channels=1024),
        )

        # Decoder part of 2DUnet model.
        # Transforming slices back up to (1,512,512) with help of the skip connections
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

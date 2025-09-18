import torch
from torch import nn

from models.TSPFM import TSPFM


class Discriminator(nn.Module):
    def __init__(self, input_channels=18):  # 12 frames input + 6 frames output
        super().__init__()

        # Initial convolution layer - Using stride 4 convolution to quickly reduce resolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            TSPFM(32, reduction_ratio=16),
            nn.PReLU()
        )

        self.high_precip_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # Depthwise separable convolution
            nn.BatchNorm2d(32),
            nn.PReLU(),
            TSPFM(32, reduction_ratio=16)
        )

        # # First CAA module - Using larger convolution kernel
        # First downsampling - From 32 channels to 64 channels
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,groups=32),
            TSPFM(64, reduction_ratio=16),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        # # Second CAA module - Medium convolution kernel
        # Second downsampling - From 64 channels to 128 channels
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=8),
            TSPFM(128, reduction_ratio=16),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,groups=64),
            TSPFM(128, reduction_ratio=16),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=16),
            TSPFM(256, reduction_ratio=16),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=128),
            TSPFM(256, reduction_ratio=16),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=32),
            TSPFM(512, reduction_ratio=16),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=256),
            TSPFM(512, reduction_ratio=16),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        # Output layer - Using Sigmoid activation function
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm = self.initial(x)
        high_precip_mask = torch.sigmoid((x >= 0.2).float())
        high_feat = self.high_precip_branch(high_precip_mask)
        x = torch.cat([norm, high_feat], 1)

        x = self.down1(x)

        x = self.down2(x)

        x = self.down3(x)

        x = self.down4(x)

        # Output discriminator result (probability value between 0-1)
        return self.output(x)
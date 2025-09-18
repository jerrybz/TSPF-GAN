import torch
from torch import nn

from models.TSPFM import TSPFM


class Discriminator(nn.Module):
    def __init__(self, input_channels=18):  # 12帧输入 + 6帧输出
        super().__init__()

        # self.high_precip_enhance = nn.Sequential(
        #     nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # 深度可分离卷积保留空间细节
        #     nn.BatchNorm2d(input_channels),
        #     nn.PReLU()
        # )
        # 初始卷积层 - 使用步长为4的卷积来快速降低分辨率
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            TSPFM(32, reduction_ratio=16),
            nn.PReLU()
        )

        self.high_precip_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 深度可分离卷积
            nn.BatchNorm2d(32),
            nn.PReLU(),
            TSPFM(32, reduction_ratio=16)
        )

        # # 第一个CAA模块 - 使用更大的卷积核
        # self.csa1 =
        # 第一个下采样 - 从32通道升到64通道
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,groups=32),
            TSPFM(64, reduction_ratio=16),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        # # 第二个CAA模块 - 中等卷积核
        # self.csa2 = CSPAM(64, reduction_ratio=16)
        # 第二个下采样 - 从64通道升到128通道
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=8),
            TSPFM(128, reduction_ratio=16),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,groups=64),
            TSPFM(128, reduction_ratio=16),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        # self.csa3 = CSPAM(128, reduction_ratio=16)

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=16),
            TSPFM(256, reduction_ratio=16),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=128),
            TSPFM(256, reduction_ratio=16),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        # self.csa4 = CSPAM(256, reduction_ratio=16)

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=32),
            TSPFM(512, reduction_ratio=16),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=256),
            TSPFM(512, reduction_ratio=16),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        # 输出层 - 使用Sigmoid激活函数
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # high_precip_mask = (x >= 0.15).float()
        # low_precip_mask = (x < 0.15).float()
        # high_precip_weight = self.high_precip_enhance(high_precip_mask)
        # x = high_precip_weight * high_precip_mask + low_precip_mask
        norm = self.initial(x)
        high_precip_mask = torch.sigmoid((x >= 0.2).float())
        high_feat = self.high_precip_branch(high_precip_mask)
        x = torch.cat([norm, high_feat], 1)
        # x = self.csa1(x)
        x = self.down1(x)

        # x = self.csa2(x)
        x = self.down2(x)
        # x = self.csa3(x)
        x = self.down3(x)
        # x = self.csa4(x)
        x = self.down4(x)

        # 输出判别结果（0-1之间的概率值）
        return self.output(x)
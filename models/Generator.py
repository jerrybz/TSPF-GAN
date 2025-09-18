from torch import nn
import torch

from models.TSPFM import TSPFM
from models.regression_lightning_GAN import Precip_regression_base
from models.unet_parts_depthwise_separable import DoubleConvDS, DownDS, UpDS, OutConv


class Generator(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        kernels_per_layer = self.hparams.kernels_per_layer
        self.bilinear = self.hparams.bilinear
        func = nn.ReLU(inplace=True)
        reduction_ratio = self.hparams.reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer, func=func)
        self.tspfm_layers = nn.ModuleDict({
            'layer1': TSPFM(64, reduction_ratio),
            'layer2': TSPFM(128, reduction_ratio),
            'layer3': TSPFM(256, reduction_ratio),
            'layer4': TSPFM(512, reduction_ratio),
            'layer5': TSPFM(1024 // (2 if self.bilinear else 1), reduction_ratio),
        })
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer, func=func)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer, func=func)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer, func=func)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer, func=func)

        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, func=func)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, func=func)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, func=func)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer, func=func)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_csp = self.tspfm_layers['layer1'](x1)

        x2 = self.down1(x1)
        x2_csp = self.tspfm_layers['layer2'](x2)

        x3 = self.down2(x2)
        x3_csp = self.tspfm_layers['layer3'](x3)

        x4 = self.down3(x3)
        x4_csp = self.tspfm_layers['layer4'](x4)

        x5 = self.down4(x4)
        x5_csp = self.tspfm_layers['layer5'](x5)

        x = self.up1(x5_csp, x4_csp)
        x = self.up2(x, x3_csp)
        x = self.up3(x, x2_csp)
        x = self.up4(x, x1_csp)
        logits = self.outc(x)

        return logits
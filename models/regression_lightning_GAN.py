import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_msssim import ssim
import argparse
import numpy as np

from utils import dataset_precip
from models.discriminator import Discriminator


class UNet_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default="UNet",
            choices=["UNet", "UNetDS", "UNet_Attention", "UNetDS_Attention"],
        )
        parser.add_argument("--n_channels", type=int, default=5)
        parser.add_argument("--n_classes", type=int, default=20)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Set manual optimization
        self.automatic_optimization = False

        # Training step counter
        self.current_step = 0


class Precip_regression_base(UNet_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNet_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=5)
        parser.add_argument("--num_output_images", type=int, default=20)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.add_argument("--lambda_gan", type=float, default=1)  # Adjust GAN loss weight
        parser.add_argument("--d_train_freq", type=int, default=2)  # Discriminator training frequency
        parser.add_argument("--g_train_freq", type=int, default=1)
        parser.add_argument("--warmup_epochs", type=int, default=1)  # Add warmup epochs
        parser.add_argument("--grad_clip_val", type=float, default=1.0)  # Gradient clipping threshold
        parser.add_argument("--max_epochs", type=int, default=200)
        parser.add_argument("--d_lr", type=float, default=1e-4)  # Discriminator learning rate
        parser.add_argument("--g_lr", type=float, default=1e-3)  # Generator learning rate
        parser.add_argument("--use_gan", type=bool, default=True)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 20
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        # Initialize discriminator
        self.discriminator = Discriminator(input_channels=25)
        # Initialize dataset related variables
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

        # Warmup counter
        self.warmup_step = 0

        # Discriminator training counter
        self.d_step = 0

    def loss_func(self, y_pred, y_true, x_input=None):
        # Pixel-level loss
        pixel_loss = nn.functional.mse_loss(y_pred, y_true, reduction='sum') / y_true.size(0)
        high_precip_mask = (y_true >= 0.15).float()
        weighted_pixel_loss = (pixel_loss * (1 + 2 * high_precip_mask)).mean()
        # SSIM loss
        ssim_loss = (1 - ssim(y_pred, y_true, data_range=1.0)).mean()

        # L1 regularization
        l1_reg = sum(p.abs().mean() for p in self.parameters())

        # Total loss
        total_loss = (
                0.7 * weighted_pixel_loss +
                0.3 * ssim_loss +
                0.0001 * l1_reg
        )

        # GAN loss
        if x_input is not None and self.training and self.hparams.use_gan:
            fake_input = torch.cat([x_input, y_pred], dim=1)
            real_input = torch.cat([x_input, y_true], dim=1)

            # Calculate heavy precipitation region mask (based on ground truth)
            high_precip_mask = (y_true >= 0.15).float().unsqueeze(1)  # 扩展为[B,1,H,W]

            # Weighted discriminator loss (heavy precipitation region loss weight × 2)
            d_real = self.discriminator(real_input)
            d_fake = self.discriminator(fake_input.detach())
            d_loss_real = nn.BCELoss(reduction='none')(d_real, torch.ones_like(d_real)) * (1 + 2 * high_precip_mask)
            d_loss_fake = nn.BCELoss(reduction='none')(d_fake, torch.zeros_like(d_fake)) * (1 + 2 * high_precip_mask)
            d_loss = (d_loss_real + d_loss_fake).mean()

            # Weighted generator loss (heavy precipitation region loss weight × 2)
            g_loss = nn.BCELoss(reduction='none')(self.discriminator(fake_input), torch.ones_like(d_fake)) * (
                    1 + 2 * high_precip_mask)
            g_loss = g_loss.mean()

            # Warmup phase
            if self.warmup_step < self.hparams.warmup_epochs:
                warmup_factor = self.warmup_step / self.hparams.warmup_epochs
                gan_weight = self.hparams.lambda_gan * warmup_factor
            else:
                gan_weight = self.hparams.lambda_gan

            # Add GAN loss
            total_loss += gan_weight * g_loss

            return total_loss, d_loss

        return total_loss, None

    def training_step(self, batch, batch_idx):
        # Get optimizers
        g_opt, d_opt = self.optimizers()

        x, y = batch
        self.current_step += 1

        # Train discriminator
        if self.current_step % self.hparams.d_train_freq == 0 and self.hparams.use_gan:
            d_opt.zero_grad()
            with torch.set_grad_enabled(True):
                y_pred = self(x)
                fake_input = torch.cat([x, y_pred], dim=1)
                real_input = torch.cat([x, y], dim=1)

                d_real = self.discriminator(real_input)
                d_fake = self.discriminator(fake_input)

                d_loss = nn.BCELoss()(d_real, torch.ones_like(d_real)) + \
                         nn.BCELoss()(d_fake, torch.zeros_like(d_fake))

                self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.manual_backward(d_loss)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.hparams.grad_clip_val)
                d_opt.step()

        # Train generator
        if self.current_step % self.hparams.g_train_freq == 0:
            y_pred = self(x)
            loss, _ = self.loss_func(y_pred, y, x)
            self.log("g_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_val)
            g_opt.step()

        # Update warmup steps
        if batch_idx == 0:
            self.warmup_step += 1

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # In validation phase, only calculate basic loss, not GAN loss
        pixel_loss = nn.functional.mse_loss(y_pred, y, reduction="sum") / y.size(0)

        self.log("val_loss", pixel_loss, prog_bar=True)

    def configure_optimizers(self):
        g_opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.g_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        g_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                g_opt,
                T_max=self.hparams.epochs,
                eta_min=self.hparams.g_lr * 0.01
            ),
            "monitor": "val_loss",
            "interval": "epoch"
        }

        # Discriminator optimizer
        d_opt = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.d_lr,
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        d_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                d_opt,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.d_lr * 0.01
            ),
            "monitor": "d_loss",
            "interval": "epoch"
        }

        return [g_opt, d_opt], [g_scheduler, d_scheduler]

    def prepare_data(self):
        train_transform = None

        valid_transform = None
        test_transform = None
        precip_dataset = (
            dataset_precip.precipitation_maps_oversampled_h5
            if self.hparams.use_oversampled_dataset
            else dataset_precip.precipitation_maps_h5
        )
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=True,
            transform=valid_transform,
        )
        self.test_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            train=False,
            transform=test_transform)

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        return test_loader

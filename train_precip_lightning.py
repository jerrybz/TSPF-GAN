import os

from models.Generator import Generator
from root import ROOT_DIR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import regression_lightning_GAN as unet_regr_g
from lightning.pytorch.tuner import Tuner


def train_regression(hparams, find_batch_size_automatically: bool = False):
    net = Generator(hparams=hparams)

    default_save_path = os.path.join(ROOT_DIR, "lightning", "precip_regression")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(default_save_path, net.__class__.__name__),
        filename=net.__class__.__name__ + "_rain_threshold_20_{epoch}-{val_loss:.6f}",
        save_top_k=hparams.topk,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)
    if "288" in hparams.model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(default_save_path, net.__class__.__name__ + "_288"),
            filename=net.__class__.__name__ + "_288" + "_rain_threshold_20_{epoch}-{val_loss:.6f}",
            save_top_k=hparams.topk,
            verbose=False,
            monitor="val_loss",
            mode="min",
        )
        lr_monitor = LearningRateMonitor()
        tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__ + "_288")
    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    callbacks = [checkpoint_callback, lr_monitor, earlystopping_callback]
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=callbacks,
        val_check_interval=hparams.val_check_interval,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(net, mode="binsearch")

    # This can be used to speed up training with newer GPUs:
    # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#low-precision-matrix-multiplication
    # torch.set_float32_matmul_precision('medium')

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = unet_regr_g.Precip_regression_base.add_model_specific_args(parser)
    parser.add_argument(
        "--dataset_folder",
        default=os.path.join(ROOT_DIR,
                             "path/to/dataset"),
        type=str,
    )
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)
    parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()

    # args.fast_dev_run = True
    args.n_channels = 5
    args.es_patience = 20
    args.gpus = 1
    args.lr_patience = 4

    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.model = "TSPF-GAN"
    print(f"Start training model: TSPF-GAN")
    train_regression(args, find_batch_size_automatically=False)

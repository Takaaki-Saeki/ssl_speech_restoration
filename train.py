import argparse
import os
import pathlib
import yaml
from dataset import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_module import (
    PretrainLightningModule,
    SSLStepLightningModule,
    SSLDualLightningModule,
)
from utils import configure_args


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=pathlib.Path)
    parser.add_argument(
        "--stage", required=True, type=str, choices=["pretrain", "ssl-step", "ssl-dual"]
    )
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--corpus_type", default=None, type=str)
    parser.add_argument("--source_path", default=None, type=pathlib.Path)
    parser.add_argument("--aux_path", default=None, type=pathlib.Path)
    parser.add_argument("--preprocessed_path", default=None, type=pathlib.Path)
    parser.add_argument("--n_train", default=None, type=int)
    parser.add_argument("--n_val", default=None, type=int)
    parser.add_argument("--n_test", default=None, type=int)
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--pretrained_path", default=None, type=pathlib.Path)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--beta", default=None, type=float)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument(
        "--feature_loss_type", default=None, type=str, choices=["mae", "mse"]
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def train(args, config, output_path):
    debug = args.debug

    csvlogger = CSVLogger(save_dir=str(output_path), name="train_log")
    tblogger = TensorBoardLogger(save_dir=str(output_path), name="tf_log")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        save_weights_only=True,
        save_top_k=-1,
        every_n_epochs=1,
        monitor="val_loss",
    )
    callbacks = [checkpoint_callback]
    if config["train"]["early_stopping"]:
        earlystop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.0, patience=15, mode="min"
        )
        callbacks.append(earlystop_callback)

    trainer = Trainer(
        max_epochs=1 if debug else config["train"]["epoch"],
        gpus=-1,
        deterministic=False,
        auto_select_gpus=True,
        benchmark=True,
        default_root_dir=os.getcwd(),
        limit_train_batches=0.01 if debug else 1.0,
        limit_val_batches=0.5 if debug else 1.0,
        callbacks=callbacks,
        logger=[csvlogger, tblogger],
        gradient_clip_val=config["train"]["grad_clip_thresh"],
        flush_logs_every_n_steps=config["train"]["logger_step"],
        val_check_interval=0.5,
    )

    if config["general"]["stage"] == "pretrain":
        model = PretrainLightningModule(config)
    elif config["general"]["stage"] == "ssl-step":
        model = SSLStepLightningModule(config)
    elif config["general"]["stage"] == "ssl-dual":
        model = SSLDualLightningModule(config)
    else:
        raise NotImplementedError()

    datamodule = DataModule(config)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":

    args = get_arg()
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    output_path = pathlib.Path(config["general"]["output_path"]) / args.run_name
    os.makedirs(output_path, exist_ok=True)

    config, args = configure_args(config, args)

    train(args=args, config=config, output_path=output_path)

import torch
import pytorch_lightning as pl
import torchaudio
import os
import pathlib
import tqdm
from model import (
    EncoderModule,
    ChannelFeatureModule,
    ChannelModule,
    MultiScaleSpectralLoss,
    GSTModule,
)
from utils import (
    manual_logging,
    load_vocoder,
    plot_and_save_mels,
    plot_and_save_mels_all,
)


class PretrainLightningModule(pl.LightningModule):
    """
    Supervised pretraining for low-resource settings.
    This module provides supervised pretraining for the analysis and channel modules.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config["general"]["use_gst"]:
            self.encoder = EncoderModule(config)
            self.gst = GSTModule(config)
        else:
            self.encoder = EncoderModule(config, use_channel=True)
            self.channelfeats = ChannelFeatureModule(config)

        self.channel = ChannelModule(config)
        self.vocoder = load_vocoder(config)

        self.criteria_a = MultiScaleSpectralLoss(config)
        if "feature_loss" in config["train"]:
            if config["train"]["feature_loss"]["type"] == "mae":
                self.criteria_b = torch.nn.L1Loss()
            else:
                self.criteria_b = torch.nn.MSELoss()
        else:
            self.criteria = torch.nn.L1Loss()
        self.alpha = config["train"]["alpha"]

    def forward(self, melspecs, wavsaux):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(melspecs.unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(melspecs.transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(melspecs.unsqueeze(1).transpose(2, 3))
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        wavsdeg = self.channel(wavsaux, chfeats)
        return enc_out, wavsdeg

    def training_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        wavsdeg = self.channel(batch["wavsaux"], chfeats)
        loss_recons = self.criteria_a(wavsdeg, batch["wavs"])
        if self.config["general"]["feature_type"] == "melspec":
            loss_encoder = self.criteria_b(enc_out, batch["melspecsaux"])
        elif self.config["general"]["feature_type"] == "vocfeats":
            loss_encoder = self.criteria_b(enc_out, batch["melceps"])
        loss = self.alpha * loss_recons + (1.0 - self.alpha) * loss_encoder
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss_recons",
            loss_recons,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_encoder",
            loss_encoder,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        wavsdeg = self.channel(batch["wavsaux"], chfeats)
        loss_recons = self.criteria_a(wavsdeg, batch["wavs"])
        if self.config["general"]["feature_type"] == "melspec":
            val_aux_feats = batch["melspecsaux"]
            feats_name = "melspec"
            loss_encoder = self.criteria_b(enc_out, val_aux_feats)
        elif self.config["general"]["feature_type"] == "vocfeats":
            val_aux_feats = batch["melceps"]
            feats_name = "melcep"
            loss_encoder = self.criteria_b(enc_out, val_aux_feats)
        loss = self.alpha * loss_recons + (1.0 - self.alpha) * loss_encoder
        logger_img_dict = {
            "val_src_melspec": batch["melspecs"],
            "val_pred_{}".format(feats_name): enc_out,
            "val_aux_{}".format(feats_name): val_aux_feats,
        }
        logger_wav_dict = {
            "val_src_wav": batch["wavs"],
            "val_pred_wav": wavsdeg,
            "val_aux_wav": batch["wavsaux"],
        }
        return {
            "val_loss": loss,
            "val_loss_recons": loss_recons,
            "val_loss_encoder": loss_encoder,
            "logger_dict": [logger_img_dict, logger_wav_dict],
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out["val_loss"] for out in outputs]).mean().item()
        val_loss_recons = (
            torch.stack([out["val_loss_recons"] for out in outputs]).mean().item()
        )
        val_loss_encoder = (
            torch.stack([out["val_loss_encoder"] for out in outputs]).mean().item()
        )
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_loss_recons",
            val_loss_recons,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_encoder",
            val_loss_encoder,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][0], data_type="image")
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][1], data_type="audio")

    def test_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        wavsdeg = self.channel(batch["wavsaux"], chfeats)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
            enc_feats_aux = batch["melspecsaux"]
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
            enc_feats_aux = torch.cat(
                (batch["f0s"].unsqueeze(1), batch["melceps"]), dim=1
            )
        recons_wav = self.vocoder(enc_feats_aux).squeeze(1)
        remas = self.vocoder(enc_feats).squeeze(1)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats_input = batch["melspecs"]
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats_input = torch.cat(
                (batch["f0s"].unsqueeze(1), batch["melcepssrc"]), dim=1
            )
        input_recons = self.vocoder(enc_feats_input).squeeze(1)
        if "wavsaux" in batch:
            gt_wav = batch["wavsaux"]
        else:
            gt_wav = None
        return {
            "reconstructed": recons_wav,
            "remastered": remas,
            "channeled": wavsdeg,
            "groundtruth": gt_wav,
            "input": batch["wavs"],
            "input_recons": input_recons,
        }

    def test_epoch_end(self, outputs):
        wav_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_wavs"
        )
        os.makedirs(wav_dir, exist_ok=True)
        mel_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_mels"
        )
        os.makedirs(mel_dir, exist_ok=True)
        print("Saving mel spectrogram plots ...")
        for idx, out in enumerate(tqdm.tqdm(outputs)):
            for key in [
                "reconstructed",
                "remastered",
                "channeled",
                "input",
                "input_recons",
                "groundtruth",
            ]:
                if out[key] != None:
                    torchaudio.save(
                        wav_dir / "{}-{}.wav".format(idx, key),
                        out[key][0, ...].unsqueeze(0).cpu(),
                        sample_rate=self.config["preprocess"]["sampling_rate"],
                        channels_first=True,
                    )
                    plot_and_save_mels(
                        out[key][0, ...].cpu(),
                        mel_dir / "{}-{}.png".format(idx, key),
                        self.config,
                    )
            plot_and_save_mels_all(
                out,
                [
                    "reconstructed",
                    "remastered",
                    "channeled",
                    "input",
                    "input_recons",
                    "groundtruth",
                ],
                mel_dir / "{}-all.png".format(idx),
                self.config,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config["train"]["learning_rate"]
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, min_lr=1e-5, verbose=True
            ),
            "interval": "epoch",
            "frequency": 3,
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def tflogger(self, logger_dict, data_type):
        for lg in self.logger.experiment:
            if type(lg).__name__ == "SummaryWriter":
                tensorboard = lg
        for key in logger_dict.keys():
            manual_logging(
                logger=tensorboard,
                item=logger_dict[key],
                idx=0,
                tag=key,
                global_step=self.global_step,
                data_type=data_type,
                config=self.config,
            )


class SSLBaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config["general"]["use_gst"]:
            self.encoder = EncoderModule(config)
            self.gst = GSTModule(config)
        else:
            self.encoder = EncoderModule(config, use_channel=True)
            self.channelfeats = ChannelFeatureModule(config)
        self.channel = ChannelModule(config)

        if config["train"]["load_pretrained"]:
            pre_model = PretrainLightningModule.load_from_checkpoint(
                checkpoint_path=config["train"]["pretrained_path"]
            )
            self.encoder.load_state_dict(pre_model.encoder.state_dict(), strict=False)
            self.channel.load_state_dict(pre_model.channel.state_dict(), strict=False)
            if config["general"]["use_gst"]:
                self.gst.load_state_dict(pre_model.gst.state_dict(), strict=False)
            else:
                self.channelfeats.load_state_dict(
                    pre_model.channelfeats.state_dict(), strict=False
                )

        self.vocoder = load_vocoder(config)
        self.criteria = self.get_loss_function(config)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def get_loss_function(self, config):
        raise NotImplementedError()

    def forward(self, melspecs, f0s=None):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(melspecs.unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(melspecs.transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(melspecs.unsqueeze(1).transpose(2, 3))
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((f0s.unsqueeze(1), enc_out), dim=1)
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        return remas, wavsdeg

    def test_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats_input = batch["melspecs"]
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats_input = torch.cat(
                (batch["f0s"].unsqueeze(1), batch["melcepssrc"]), dim=1
            )
        input_recons = self.vocoder(enc_feats_input).squeeze(1)
        if "wavsaux" in batch:
            gt_wav = batch["wavsaux"]
            if self.config["general"]["feature_type"] == "melspec":
                enc_feats_aux = batch["melspecsaux"]
            elif self.config["general"]["feature_type"] == "vocfeats":
                enc_feats_aux = torch.cat(
                    (batch["f0s"].unsqueeze(1), batch["melceps"]), dim=1
                )
            recons_wav = self.vocoder(enc_feats_aux).squeeze(1)
        else:
            gt_wav = None
            recons_wav = None
        return {
            "reconstructed": recons_wav,
            "remastered": remas,
            "channeled": wavsdeg,
            "input": batch["wavs"],
            "input_recons": input_recons,
            "groundtruth": gt_wav,
        }

    def test_epoch_end(self, outputs):
        wav_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_wavs"
        )
        os.makedirs(wav_dir, exist_ok=True)
        mel_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_mels"
        )
        os.makedirs(mel_dir, exist_ok=True)
        print("Saving mel spectrogram plots ...")
        for idx, out in enumerate(tqdm.tqdm(outputs)):
            plot_keys = []
            for key in [
                "reconstructed",
                "remastered",
                "channeled",
                "input",
                "input_recons",
                "groundtruth",
            ]:
                if out[key] != None:
                    plot_keys.append(key)
                    torchaudio.save(
                        wav_dir / "{}-{}.wav".format(idx, key),
                        out[key][0, ...].unsqueeze(0).cpu(),
                        sample_rate=self.config["preprocess"]["sampling_rate"],
                        channels_first=True,
                    )
                    plot_and_save_mels(
                        out[key][0, ...].cpu(),
                        mel_dir / "{}-{}.png".format(idx, key),
                        self.config,
                    )
            plot_and_save_mels_all(
                out,
                plot_keys,
                mel_dir / "{}-all.png".format(idx),
                self.config,
            )

    def tflogger(self, logger_dict, data_type):
        for lg in self.logger.experiment:
            if type(lg).__name__ == "SummaryWriter":
                tensorboard = lg
        for key in logger_dict.keys():
            manual_logging(
                logger=tensorboard,
                item=logger_dict[key],
                idx=0,
                tag=key,
                global_step=self.global_step,
                data_type=data_type,
                config=self.config,
            )


class SSLStepLightningModule(SSLBaseModule):
    """
    Self-supervised speech restoration model
    with fine-tuning supervisedly pretrained model,
    correspond to ``SSL-pre'' in the paper.

    This module provises step-wise learning, which only trains the channel module at early epochs
    and then only train the analysis module ar later epochs to stabilize the traininig.
    """

    def __init__(self, config):
        super().__init__(config)
        if config["train"]["fix_channel"]:
            for param in self.channel.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        loss = self.criteria(wavsdeg, batch["wavs"])
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
            feats_name = "melspec"
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
            feats_name = "melcep"
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        loss = self.criteria(wavsdeg, batch["wavs"])
        logger_img_dict = {
            "val_src_melspec": batch["melspecs"],
            "val_pred_{}".format(feats_name): enc_out,
        }
        for auxfeats in ["melceps", "melspecsaux"]:
            if auxfeats in batch:
                logger_img_dict["val_aux_{}".format(auxfeats)] = batch[auxfeats]
        logger_wav_dict = {
            "val_src_wav": batch["wavs"],
            "val_remastered_wav": remas,
            "val_pred_wav": wavsdeg,
        }
        if "wavsaux" in batch:
            logger_wav_dict["val_aux_wav"] = batch["wavsaux"]
        d_out = {"val_loss": loss, "logger_dict": [logger_img_dict, logger_wav_dict]}
        return d_out

    def validation_epoch_end(self, outputs):
        self.log(
            "val_loss",
            torch.stack([out["val_loss"] for out in outputs]).mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][0], data_type="image")
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][1], data_type="audio")

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if epoch < self.config["train"]["epoch_channel"]:
            if optimizer_idx == 0:
                optimizer.step(closure=optimizer_closure)
            elif optimizer_idx == 1:
                optimizer_closure()
        else:
            if optimizer_idx == 0:
                optimizer_closure()
            elif optimizer_idx == 1:
                optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        if self.config["train"]["fix_channel"]:
            if self.config["general"]["use_gst"]:
                optimizer_channel = torch.optim.Adam(
                    self.gst.parameters(), lr=self.config["train"]["learning_rate"]
                )
            else:
                optimizer_channel = torch.optim.Adam(
                    self.channelfeats.parameters(),
                    lr=self.config["train"]["learning_rate"],
                )
            optimizer_encoder = torch.optim.Adam(
                self.encoder.parameters(), lr=self.config["train"]["learning_rate"]
            )
        else:
            if self.config["general"]["use_gst"]:
                optimizer_channel = torch.optim.Adam(
                    [
                        {"params": self.channel.parameters()},
                        {"params": self.gst.parameters()},
                    ],
                    lr=self.config["train"]["learning_rate"],
                )
            else:
                optimizer_channel = torch.optim.Adam(
                    [
                        {"params": self.channel.parameters()},
                        {"params": self.channelfeats.parameters()},
                    ],
                    lr=self.config["train"]["learning_rate"],
                )
            optimizer_encoder = torch.optim.Adam(
                self.encoder.parameters(), lr=self.config["train"]["learning_rate"]
            )
        optimizers = [optimizer_channel, optimizer_encoder]
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0], mode="min", factor=0.5, min_lr=1e-5, verbose=True
                ),
                "interval": "epoch",
                "frequency": 3,
                "monitor": "val_loss",
            },
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[1], mode="min", factor=0.5, min_lr=1e-5, verbose=True
                ),
                "interval": "epoch",
                "frequency": 3,
                "monitor": "val_loss",
            },
        ]
        return optimizers, schedulers

    def get_loss_function(self, config):
        return MultiScaleSpectralLoss(config)


class SSLDualLightningModule(SSLBaseModule):
    """
    Self-supervised speech restoration model with dual learning,
    correspond to ``SSL-dual'' or ``SSL-dual-pre'' in the paper.

    This module provises dual-learning.
    In addition to the basic training framework, we introduce a training task that
    propagates information in the reverse direction.
    """

    def __init__(self, config):
        super().__init__(config)
        if config["train"]["fix_channel"]:
            for param in self.channel.parameters():
                param.requires_grad = False
        self.spec_module = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["preprocess"]["sampling_rate"],
            n_fft=config["preprocess"]["fft_length"],
            win_length=config["preprocess"]["frame_length"],
            hop_length=config["preprocess"]["frame_shift"],
            f_min=config["preprocess"]["fmin"],
            f_max=config["preprocess"]["fmax"],
            n_mels=config["preprocess"]["n_mels"],
            power=1,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )
        self.beta = config["train"]["beta"]
        self.criteria_a, self.criteria_b = self.get_loss_function(config)

    def training_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        loss_recons = self.criteria_a(wavsdeg, batch["wavs"])

        with torch.no_grad():
            wavsdegtask = self.channel(batch["wavstask"], chfeats)
        melspecstask = self.calc_spectrogram(wavsdegtask)
        if self.config["general"]["use_gst"]:
            enc_out_task = self.encoder(melspecstask.unsqueeze(1).transpose(2, 3))
        else:
            enc_out_task, _ = self.encoder(melspecstask.unsqueeze(1).transpose(2, 3))
        enc_out_task = enc_out_task.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            loss_task = self.criteria_b(enc_out_task, batch["melspecstask"])
        elif self.config["general"]["feature_type"] == "vocfeats":
            loss_task = self.criteria_b(enc_out_task, batch["melcepstask"])
        loss = self.beta * loss_recons + (1 - self.beta) * loss_task

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss_recons",
            loss_recons,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_task",
            loss_task,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
            feats_name = "melspec"
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
            feats_name = "melcep"
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        loss_recons = self.criteria_a(wavsdeg, batch["wavs"])

        wavsdegtask = self.channel(batch["wavstask"], chfeats)
        melspecstask = self.calc_spectrogram(wavsdegtask)
        if self.config["general"]["use_gst"]:
            enc_out_task = self.encoder(melspecstask.unsqueeze(1).transpose(2, 3))
        else:
            enc_out_task, _ = self.encoder(melspecstask.unsqueeze(1).transpose(2, 3))
        enc_out_task = enc_out_task.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_out_task_truth = batch["melspecstask"]
            loss_task = self.criteria_b(enc_out_task, enc_out_task_truth)
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_out_task_truth = batch["melcepstask"]
            loss_task = self.criteria_b(enc_out_task, enc_out_task_truth)
        loss = self.beta * loss_recons + (1 - self.beta) * loss_task

        logger_img_dict = {
            "val_src_melspec": batch["melspecs"],
            "val_pred_{}".format(feats_name): enc_out,
            "val_truth_{}_task".format(feats_name): enc_out_task_truth,
            "val_pred_{}_task".format(feats_name): enc_out_task,
        }
        for auxfeats in ["melceps", "melspecsaux"]:
            if auxfeats in batch:
                logger_img_dict["val_aux_{}".format(auxfeats)] = batch[auxfeats]
        logger_wav_dict = {
            "val_src_wav": batch["wavs"],
            "val_remastered_wav": remas,
            "val_pred_wav": wavsdeg,
            "val_truth_wavtask": batch["wavstask"],
            "val_deg_wavtask": wavsdegtask,
        }
        if "wavsaux" in batch:
            logger_wav_dict["val_aux_wav"] = batch["wavsaux"]

        d_out = {
            "val_loss": loss,
            "val_loss_recons": loss_recons,
            "val_loss_task": loss_task,
            "logger_dict": [logger_img_dict, logger_wav_dict],
        }
        return d_out

    def validation_epoch_end(self, outputs):
        self.log(
            "val_loss",
            torch.stack([out["val_loss"] for out in outputs]).mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_recons",
            torch.stack([out["val_loss_recons"] for out in outputs]).mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_task",
            torch.stack([out["val_loss_task"] for out in outputs]).mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][0], data_type="image")
        self.tflogger(logger_dict=outputs[-1]["logger_dict"][1], data_type="audio")

    def test_step(self, batch, batch_idx):
        if self.config["general"]["use_gst"]:
            enc_out = self.encoder(batch["melspecs"].unsqueeze(1).transpose(2, 3))
            chfeats = self.gst(batch["melspecs"].transpose(1, 2))
        else:
            enc_out, enc_hidden = self.encoder(
                batch["melspecs"].unsqueeze(1).transpose(2, 3)
            )
            chfeats = self.channelfeats(enc_hidden)
        enc_out = enc_out.squeeze(1).transpose(1, 2)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats = enc_out
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats = torch.cat((batch["f0s"].unsqueeze(1), enc_out), dim=1)
        remas = self.vocoder(enc_feats).squeeze(1)
        wavsdeg = self.channel(remas, chfeats)
        if self.config["general"]["feature_type"] == "melspec":
            enc_feats_input = batch["melspecs"]
        elif self.config["general"]["feature_type"] == "vocfeats":
            enc_feats_input = torch.cat(
                (batch["f0s"].unsqueeze(1), batch["melcepssrc"]), dim=1
            )
        input_recons = self.vocoder(enc_feats_input).squeeze(1)

        wavsdegtask = self.channel(batch["wavstask"], chfeats)
        if "wavsaux" in batch:
            gt_wav = batch["wavsaux"]
            if self.config["general"]["feature_type"] == "melspec":
                enc_feats_aux = batch["melspecsaux"]
            elif self.config["general"]["feature_type"] == "vocfeats":
                enc_feats_aux = torch.cat(
                    (batch["f0s"].unsqueeze(1), batch["melceps"]), dim=1
                )
            recons_wav = self.vocoder(enc_feats_aux).squeeze(1)
        else:
            gt_wav = None
            recons_wav = None
        return {
            "reconstructed": recons_wav,
            "remastered": remas,
            "channeled": wavsdeg,
            "channeled_task": wavsdegtask,
            "input": batch["wavs"],
            "input_recons": input_recons,
            "groundtruth": gt_wav,
        }

    def test_epoch_end(self, outputs):
        wav_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_wavs"
        )
        os.makedirs(wav_dir, exist_ok=True)
        mel_dir = (
            pathlib.Path(self.logger.experiment[0].log_dir).parent.parent / "test_mels"
        )
        os.makedirs(mel_dir, exist_ok=True)
        print("Saving mel spectrogram plots ...")
        for idx, out in enumerate(tqdm.tqdm(outputs)):
            plot_keys = []
            for key in [
                "reconstructed",
                "remastered",
                "channeled",
                "channeled_task",
                "input",
                "input_recons",
                "groundtruth",
            ]:
                if out[key] != None:
                    plot_keys.append(key)
                    torchaudio.save(
                        wav_dir / "{}-{}.wav".format(idx, key),
                        out[key][0, ...].unsqueeze(0).cpu(),
                        sample_rate=self.config["preprocess"]["sampling_rate"],
                        channels_first=True,
                    )
                    plot_and_save_mels(
                        out[key][0, ...].cpu(),
                        mel_dir / "{}-{}.png".format(idx, key),
                        self.config,
                    )
            plot_and_save_mels_all(
                out,
                plot_keys,
                mel_dir / "{}-all.png".format(idx),
                self.config,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config["train"]["learning_rate"]
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, min_lr=1e-5, verbose=True
            ),
            "interval": "epoch",
            "frequency": 3,
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def calc_spectrogram(self, wav):
        specs = self.spec_module(wav)
        log_spec = torch.log(
            torch.clamp_min(specs, self.config["preprocess"]["min_magnitude"])
            * self.config["preprocess"]["comp_factor"]
        ).to(torch.float32)
        return log_spec

    def get_loss_function(self, config):
        if config["train"]["feature_loss"]["type"] == "mae":
            feature_loss = torch.nn.L1Loss()
        else:
            feature_loss = torch.nn.MSELoss()
        return MultiScaleSpectralLoss(config), feature_loss

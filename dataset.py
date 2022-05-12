import pickle
import pathlib
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
import numpy as np
import yaml
import torchaudio
import pyworld
import pysptk
import random


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batchsize = config["train"]["batchsize"]
        self.preprocessed_dir = pathlib.Path(config["general"]["preprocessed_path"])

    def setup(self, stage):

        if not self.preprocessed_dir.exists():
            raise RuntimeError("Preprocessed directory was not be found")

        if "dual" in self.config:
            if self.config["dual"]["enable"]:
                task_config = yaml.load(
                    open(self.config["dual"]["config_path"], "r"),
                    Loader=yaml.FullLoader,
                )
                task_preprocessed_dir = (
                    self.preprocessed_dir.parent
                    / pathlib.Path(task_config["general"]["preprocessed_path"]).name
                )
                if not task_preprocessed_dir.exists():
                    raise RuntimeError(
                        "Preprocessed directory for multi-task learning was not be found"
                    )

        self.flnames = {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        }

    def get_ds(self, phase):
        ds = Dataset(self.flnames[phase], self.config)
        return ds

    def get_loader(self, phase):
        ds = self.get_ds(phase)
        dl = DataLoader(
            ds,
            self.batchsize,
            shuffle=True if phase == "train" else False,
            num_workers=self.config["train"]["num_workers"],
            drop_last=True,
        )
        return dl

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def val_dataloader(self):
        return self.get_loader(phase="val")

    def test_dataloader(self):
        return self.get_loader(phase="test")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filetxt, config):

        self.preprocessed_dir = pathlib.Path(config["general"]["preprocessed_path"])
        self.config = config
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
        self.resample_candidate = [8000, 11025, 12000, 16000]
        self.quantization_candidate = range(2 ** 6, 2 ** 10 + 2, 2)
        self.segment_length = config["preprocess"]["segment_length"]

        with open(self.preprocessed_dir / filetxt, "r") as fr:
            self.filelist = [pathlib.Path(path.strip("\n")) for path in fr]

        self.d_out = dict()
        for item in ["wavs", "wavsaux"]:
            self.d_out[item] = []

        for wp in self.filelist:

            if config["general"]["corpus_type"] == "single":
                basename = str(wp.stem)
            else:
                basename = str(wp.parent.name) + "-" + str(wp.stem)

            with open(self.preprocessed_dir / "{}.pickle".format(basename), "rb") as fw:
                d_preprocessed = pickle.load(fw)

            for item in ["wavs", "wavsaux"]:
                try:
                    self.d_out[item].extend(d_preprocessed[item])
                except:
                    pass

        for item in ["wavs", "wavsaux"]:
            if self.d_out[item] != None:
                self.d_out[item] = np.asarray(self.d_out[item])

        if "dual" in self.config:
            if self.config["dual"]["enable"]:
                task_config = yaml.load(
                    open(config["dual"]["config_path"], "r"),
                    Loader=yaml.FullLoader,
                )
                task_preprocessed_dir = (
                    self.preprocessed_dir.parent
                    / pathlib.Path(task_config["general"]["preprocessed_path"]).name
                )
                with open(task_preprocessed_dir / filetxt, "r") as fr:
                    task_filelist = [pathlib.Path(path.strip("\n")) for path in fr]
                self.d_out["wavstask"] = []
                for wp in task_filelist:
                    if task_config["general"]["corpus_type"] == "single":
                        basename = str(wp.stem)
                    else:
                        basename = str(wp.parent.name) + "-" + str(wp.stem)
                    with open(
                        task_preprocessed_dir / "{}.pickle".format(basename), "rb"
                    ) as fw:
                        d_preprocessed = pickle.load(fw)
                    self.d_out["wavstask"].extend(d_preprocessed["wavs"])
                self.d_out["wavstask"] = np.asarray(self.d_out["wavstask"])

    def __len__(self):
        return len(self.d_out["wavs"])

    def __getitem__(self, idx):

        d_batch = {}

        if self.d_out["wavs"].size > 0:
            d_batch["wavs"] = torch.from_numpy(self.d_out["wavs"][idx])

        if self.d_out["wavsaux"].size > 0:
            d_batch["wavsaux"] = torch.from_numpy(self.d_out["wavsaux"][idx])
        
        if (self.d_out["wavs"].size > 0) & (self.segment_length > 0):
            if self.d_out["wavsaux"].size > 0:
                d_batch["wavs"], d_batch["wavsaux"] = self.get_segment(
                    d_batch["wavs"],
                    self.segment_length,
                    d_batch["wavsaux"]
                )
            else:
                d_batch["wavs"] = self.get_segment(d_batch["wavs"], self.segment_length)

        if self.config["general"]["stage"] == "pretrain":
            if self.config["train"]["augment"]:
                d_batch["wavs"] = self.augmentation(d_batch["wavsaux"])
            d_batch["wavs"] = self.normalize_waveform(d_batch["wavs"], db=-3)
            d_batch["wavsaux"] = self.normalize_waveform(d_batch["wavsaux"], db=-3)
            if len(d_batch["wavs"]) != len(d_batch["wavsaux"]):
                min_seq_len = min(len(d_batch["wavs"]), len(d_batch["wavsaux"]))
                d_batch["wavs"] = d_batch["wavs"][:min_seq_len]
                d_batch["wavsaux"] = d_batch["wavsaux"][:min_seq_len]
            d_batch["melspecs"] = self.calc_spectrogram(d_batch["wavs"])
            if self.config["general"]["feature_type"] == "melspec":
                d_batch["melspecsaux"] = self.calc_spectrogram(d_batch["wavsaux"])
            elif self.config["general"]["feature_type"] == "vocfeats":
                d_batch["melceps"] = self.calc_melcep(d_batch["wavsaux"])
                d_batch["f0s"] = self.calc_f0(d_batch["wavs"])
                d_batch["melcepssrc"] = self.calc_melcep(d_batch["wavs"])
            else:
                raise NotImplementedError()

        elif self.config["general"]["stage"].startswith("ssl"):
            d_batch["wavs"] = self.normalize_waveform(d_batch["wavs"], db=-3)
            d_batch["melspecs"] = self.calc_spectrogram(d_batch["wavs"])
            if self.config["general"]["feature_type"] == "vocfeats":
                d_batch["f0s"] = self.calc_f0(d_batch["wavs"])
                d_batch["melcepssrc"] = self.calc_melcep(d_batch["wavs"])
            if self.d_out["wavsaux"].size > 0:
                d_batch["wavsaux"] = self.normalize_waveform(d_batch["wavsaux"], db=-3)
                if self.config["general"]["feature_type"] == "melspec":
                    d_batch["melspecsaux"] = self.calc_spectrogram(d_batch["wavsaux"])
                elif self.config["general"]["feature_type"] == "vocfeats":
                    d_batch["melceps"] = self.calc_melcep(d_batch["wavsaux"])
            if "dual" in self.config:
                if self.config["dual"]["enable"]:
                    d_batch["wavstask"] = torch.from_numpy(self.d_out["wavstask"][idx])
                    if self.segment_length > 0:
                        d_batch["wavstask"] = self.get_segment(
                            d_batch["wavstask"], self.segment_length
                        )
                    d_batch["wavstask"] = self.normalize_waveform(
                        d_batch["wavstask"], db=-3
                    )
                    if self.config["general"]["feature_type"] == "melspec":
                        d_batch["melspecstask"] = self.calc_spectrogram(
                            d_batch["wavstask"]
                        )
                    elif self.config["general"]["feature_type"] == "vocfeats":
                        d_batch["melcepstask"] = self.calc_melcep(d_batch["wavstask"])
                    else:
                        raise NotImplementedError()
        else:
            raise NotImplementedError()

        return d_batch

    def calc_spectrogram(self, wav):
        specs = self.spec_module(wav)
        log_spec = torch.log(
            torch.clamp_min(specs, self.config["preprocess"]["min_magnitude"])
            * self.config["preprocess"]["comp_factor"]
        ).to(torch.float32)
        return log_spec

    def calc_melcep(self, wav):
        wav = wav.numpy()
        _, sp, _ = pyworld.wav2world(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            fft_size=self.config["preprocess"]["fft_length"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        melcep = pysptk.sp2mc(
            sp,
            order=self.config["preprocess"]["cep_order"],
            alpha=pysptk.util.mcepalpha(self.config["preprocess"]["sampling_rate"]),
        ).transpose(1, 0)
        melcep = torch.from_numpy(melcep).to(torch.float32)
        return melcep

    def calc_f0(self, wav):
        if self.config["preprocess"]["f0_extractor"] == "dio":
            return self.calc_f0_dio(wav)
        elif self.config["preprocess"]["f0_extractor"] == "harvest":
            return self.calc_f0_harvest(wav)
        elif self.config["preprocess"]["f0_extractor"] == "swipe":
            return self.calc_f0_swipe(wav)
        else:
            raise NotImplementedError()

    def calc_f0_dio(self, wav):
        wav = wav.numpy()
        _f0, _t = pyworld.dio(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        f0 = pyworld.stonemask(
            wav.astype(np.float64), _f0, _t, self.config["preprocess"]["sampling_rate"]
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def calc_f0_harvest(self, wav):
        wav = wav.numpy()
        _f0, _t = pyworld.harvest(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        f0 = pyworld.stonemask(
            wav.astype(np.float64), _f0, _t, self.config["preprocess"]["sampling_rate"]
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def calc_f0_swipe(self, wav):
        wav = wav.numpy()
        f0 = pysptk.sptk.swipe(
            wav.astype(np.float64),
            fs=self.config["preprocess"]["sampling_rate"],
            min=71,
            max=800,
            hopsize=self.config["preprocess"]["frame_shift"],
            otype="f0",
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def augmentation(self, wav):
        wav /= torch.max(torch.abs(wav))
        new_freq = random.choice(self.resample_candidate)
        new_quantization = random.choice(self.quantization_candidate)
        mulaw_encoder = torchaudio.transforms.MuLawEncoding(
            quantization_channels=new_quantization
        )
        wav_quantized = mulaw_encoder(wav) / new_quantization * 2.0 - 1.0
        downsampler = torchaudio.transforms.Resample(
            orig_freq=self.config["preprocess"]["sampling_rate"],
            new_freq=new_freq,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )
        upsampler = torchaudio.transforms.Resample(
            orig_freq=new_freq,
            new_freq=self.config["preprocess"]["sampling_rate"],
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )
        wav_processed = upsampler(downsampler(wav_quantized))
        return wav_processed

    def normalize_waveform(self, wav, db=-3):
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav.unsqueeze(0),
            self.config["preprocess"]["sampling_rate"],
            [["norm", "{}".format(db)]],
        )
        return wav.squeeze(0)

    def get_segment(self, wav, segment_length, wavaux=None):
        seg_size = self.config["preprocess"]["sampling_rate"] * segment_length
        if len(wav) >= seg_size:
            max_wav_start = len(wav) - seg_size
            wav_start = random.randint(0, max_wav_start)
            wav = wav[wav_start : wav_start + seg_size]
            if wavaux != None:
                wavaux = wavaux[wav_start : wav_start + seg_size]
        else:
            wav = torch.nn.functional.pad(wav, (0, seg_size - len(wav)), "constant")
            if wavaux != None:
                wavaux = torch.nn.functional.pad(wavaux, (0, seg_size - len(wavaux)), "constant")
        if wavaux != None:
            return wav, wavaux
        else:
            return wav

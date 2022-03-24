import argparse
import pathlib
import yaml
import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import random
import librosa
from dataset import Dataset
import pickle
from lightning_module import (
    SSLStepLightningModule,
    SSLDualLightningModule,
)
from utils import plot_and_save_mels
import os
import tqdm


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=str)
    parser.add_argument("--config_path", required=True, type=pathlib.Path)
    parser.add_argument("--exist_src_aux", action="store_true")
    parser.add_argument("--run_name", required=True, type=str)
    return parser.parse_args()


class AETDataset(Dataset):
    def __init__(self, filetxt, src_config, tar_config):
        self.config = src_config

        self.preprocessed_dir_src = pathlib.Path(
            src_config["general"]["preprocessed_path"]
        )
        self.preprocessed_dir_tar = pathlib.Path(
            tar_config["general"]["preprocessed_path"]
        )
        for item in [
            "sampling_rate",
            "fft_length",
            "frame_length",
            "frame_shift",
            "fmin",
            "fmax",
            "n_mels",
        ]:
            assert src_config["preprocess"][item] == tar_config["preprocess"][item]

        self.spec_module = torchaudio.transforms.MelSpectrogram(
            sample_rate=src_config["preprocess"]["sampling_rate"],
            n_fft=src_config["preprocess"]["fft_length"],
            win_length=src_config["preprocess"]["frame_length"],
            hop_length=src_config["preprocess"]["frame_shift"],
            f_min=src_config["preprocess"]["fmin"],
            f_max=src_config["preprocess"]["fmax"],
            n_mels=src_config["preprocess"]["n_mels"],
            power=1,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

        with open(self.preprocessed_dir_src / filetxt, "r") as fr:
            self.filelist_src = [pathlib.Path(path.strip("\n")) for path in fr]
        with open(self.preprocessed_dir_tar / filetxt, "r") as fr:
            self.filelist_tar = [pathlib.Path(path.strip("\n")) for path in fr]

        self.d_out = {"src": {}, "tar": {}}
        for item in ["wavs", "wavsaux"]:
            self.d_out["src"][item] = []
            self.d_out["tar"][item] = []

        for swp in self.filelist_src:
            if src_config["general"]["corpus_type"] == "single":
                basename = str(swp.stem)
            else:
                basename = str(swp.parent.name) + "-" + str(swp.stem)
            with open(
                self.preprocessed_dir_src / "{}.pickle".format(basename), "rb"
            ) as fw:
                d_preprocessed = pickle.load(fw)
            for item in ["wavs", "wavsaux"]:
                try:
                    self.d_out["src"][item].extend(d_preprocessed[item])
                except:
                    pass

        for twp in self.filelist_tar:
            if tar_config["general"]["corpus_type"] == "single":
                basename = str(twp.stem)
            else:
                basename = str(twp.parent.name) + "-" + str(twp.stem)
            with open(
                self.preprocessed_dir_tar / "{}.pickle".format(basename), "rb"
            ) as fw:
                d_preprocessed = pickle.load(fw)
            for item in ["wavs", "wavsaux"]:
                try:
                    self.d_out["tar"][item].extend(d_preprocessed[item])
                except:
                    pass

        min_len = min(len(self.d_out["src"]["wavs"]), len(self.d_out["tar"]["wavs"]))
        for spk in ["src", "tar"]:
            for item in ["wavs", "wavsaux"]:
                if self.d_out[spk][item] != None:
                    self.d_out[spk][item] = np.asarray(self.d_out[spk][item][:min_len])

    def __len__(self):
        return len(self.d_out["src"]["wavs"])

    def __getitem__(self, idx):
        d_batch = {}

        for spk in ["src", "tar"]:
            for item in ["wavs", "wavsaux"]:
                if self.d_out[spk][item].size > 0:
                    d_batch["{}_{}".format(item, spk)] = torch.from_numpy(
                        self.d_out[spk][item][idx]
                    )
                    d_batch["{}_{}".format(item, spk)] = self.normalize_waveform(
                        d_batch["{}_{}".format(item, spk)], db=-3
                    )

        d_batch["melspecs_src"] = self.calc_spectrogram(d_batch["wavs_src"])
        return d_batch


class AETModule(torch.nn.Module):
    """
    src: Dataset from which we extract the channel features
    tar: Dataset to which the src channel features are added
    """

    def __init__(self, args, chmatch_config, src_config, tar_config):
        super().__init__()
        if args.stage == "ssl-step":
            LModule = SSLStepLightningModule
        elif args.stage == "ssl-dual":
            LModule = SSLDualLightningModule
        else:
            raise NotImplementedError()

        src_model = LModule(src_config).load_from_checkpoint(
            checkpoint_path=chmatch_config["general"]["source"]["ckpt_path"],
            config=src_config,
        )
        self.src_config = src_config

        self.encoder_src = src_model.encoder
        if src_config["general"]["use_gst"]:
            self.gst_src = src_model.gst
        else:
            self.channelfeats_src = src_model.channelfeats
        self.channel_src = src_model.channel

    def forward(self, melspecs_src, wavsaux_tar):
        if self.src_config["general"]["use_gst"]:
            chfeats_src = self.gst_src(melspecs_src.transpose(1, 2))
        else:
            _, enc_hidden_src = self.encoder_src(
                melspecs_src.unsqueeze(1).transpose(2, 3)
            )
            chfeats_src = self.channelfeats_src(enc_hidden_src)
        wavschmatch_tar = self.channel_src(wavsaux_tar, chfeats_src)
        return wavschmatch_tar


def calc_deg_baseline(wav, char_vector, tar_config):
    wav = wav[0, ...].cpu().detach().numpy()
    spec = librosa.stft(
        wav,
        n_fft=tar_config["preprocess"]["fft_length"],
        hop_length=tar_config["preprocess"]["frame_shift"],
        win_length=tar_config["preprocess"]["frame_length"],
    )
    spec_converted = spec * char_vector.reshape(-1, 1)
    wav_converted = librosa.istft(
        spec_converted,
        hop_length=tar_config["preprocess"]["frame_shift"],
        win_length=tar_config["preprocess"]["frame_length"],
    )
    wav_converted = torch.from_numpy(wav_converted).to(torch.float32).unsqueeze(0)
    return wav_converted


def calc_deg_charactaristics(chmatch_config):
    src_config = yaml.load(
        open(chmatch_config["general"]["source"]["config_path"], "r"),
        Loader=yaml.FullLoader,
    )
    tar_config = yaml.load(
        open(chmatch_config["general"]["target"]["config_path"], "r"),
        Loader=yaml.FullLoader,
    )
    # configs
    preprocessed_dir = pathlib.Path(src_config["general"]["preprocessed_path"])
    n_train = src_config["preprocess"]["n_train"]
    SR = src_config["preprocess"]["sampling_rate"]

    os.makedirs(preprocessed_dir, exist_ok=True)

    sourcepath = pathlib.Path(src_config["general"]["source_path"])

    if src_config["general"]["corpus_type"] == "single":
        fulllist = list(sourcepath.glob("*.wav"))
        random.seed(0)
        random.shuffle(fulllist)
        train_filelist = fulllist[:n_train]
    elif src_config["general"]["corpus_type"] == "multi-seen":
        fulllist = list(sourcepath.glob("*/*.wav"))
        random.seed(0)
        random.shuffle(fulllist)
        train_filelist = fulllist[:n_train]
    elif src_config["general"]["corpus_type"] == "multi-unseen":
        spk_list = list(set([x.parent for x in sourcepath.glob("*/*.wav")]))
        train_filelist = []
        random.seed(0)
        random.shuffle(spk_list)
        for i, spk in enumerate(spk_list):
            sourcespkpath = sourcepath / spk
            if i < n_train:
                train_filelist.extend(list(sourcespkpath.glob("*.wav")))
    else:
        raise NotImplementedError(
            "corpus_type specified in config.yaml should be {single, multi-seen, multi-unseen}"
        )

    specs_all = np.zeros((tar_config["preprocess"]["fft_length"] // 2 + 1, 1))

    for wp in tqdm.tqdm(train_filelist):
        wav, _ = librosa.load(wp, sr=SR)
        spec = np.abs(
            librosa.stft(
                wav,
                n_fft=src_config["preprocess"]["fft_length"],
                hop_length=src_config["preprocess"]["frame_shift"],
                win_length=src_config["preprocess"]["frame_length"],
            )
        )

        auxpath = pathlib.Path(src_config["general"]["aux_path"])
        if src_config["general"]["corpus_type"] == "single":
            wav_aux, _ = librosa.load(auxpath / wp.name, sr=SR)
        else:
            wav_aux, _ = librosa.load(auxpath / wp.parent.name / wp.name, sr=SR)
        spec_aux = np.abs(
            librosa.stft(
                wav_aux,
                n_fft=src_config["preprocess"]["fft_length"],
                hop_length=src_config["preprocess"]["frame_shift"],
                win_length=src_config["preprocess"]["frame_length"],
            )
        )
        min_len = min(spec.shape[1], spec_aux.shape[1])
        spec_diff = spec[:, :min_len] / (spec_aux[:, :min_len] + 1e-10)
        specs_all = np.hstack([specs_all, np.mean(spec_diff, axis=1).reshape(-1, 1)])

    char_vector = np.mean(specs_all, axis=1)
    char_vector = char_vector / (np.sum(char_vector) + 1e-10)
    return char_vector


def normalize_waveform(wav, tar_config, db=-3):
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav,
        tar_config["preprocess"]["sampling_rate"],
        [["norm", "{}".format(db)]],
    )
    return wav


def main(args, chmatch_config, device):
    src_config = yaml.load(
        open(chmatch_config["general"]["source"]["config_path"], "r"),
        Loader=yaml.FullLoader,
    )
    tar_config = yaml.load(
        open(chmatch_config["general"]["target"]["config_path"], "r"),
        Loader=yaml.FullLoader,
    )
    output_path = pathlib.Path(chmatch_config["general"]["output_path"]) / args.run_name
    dataset = AETDataset("test.txt", src_config, tar_config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    chmatch_module = AETModule(args, chmatch_config, src_config, tar_config).to(device)

    if args.exist_src_aux:
        char_vector = calc_deg_charactaristics(chmatch_config)

    for idx, batch in enumerate(tqdm.tqdm(loader)):
        melspecs_src = batch["melspecs_src"].to(device)
        wavsdeg_src = batch["wavs_src"].to(device)
        wavsaux_tar = batch["wavsaux_tar"].to(device)
        if args.exist_src_aux:
            wavsdegbaseline_tar = calc_deg_baseline(
                batch["wavsaux_tar"], char_vector, tar_config
            )
            wavsdegbaseline_tar = normalize_waveform(wavsdegbaseline_tar, tar_config)
            wavsdeg_tar = batch["wavs_tar"].to(device)
        wavsmatch_tar = normalize_waveform(
            chmatch_module(melspecs_src, wavsaux_tar).cpu().detach(), tar_config
        )
        torchaudio.save(
            output_path / "test_wavs" / "{}-src_wavsdeg.wav".format(idx),
            wavsdeg_src.cpu(),
            src_config["preprocess"]["sampling_rate"],
        )
        torchaudio.save(
            output_path / "test_wavs" / "{}-tar_wavsaux.wav".format(idx),
            wavsaux_tar.cpu(),
            tar_config["preprocess"]["sampling_rate"],
        )
        if args.exist_src_aux:
            torchaudio.save(
                output_path / "test_wavs" / "{}-tar_wavsdegbaseline.wav".format(idx),
                wavsdegbaseline_tar.cpu(),
                tar_config["preprocess"]["sampling_rate"],
            )
            torchaudio.save(
                output_path / "test_wavs" / "{}-tar_wavsdeg.wav".format(idx),
                wavsdeg_tar.cpu(),
                tar_config["preprocess"]["sampling_rate"],
            )
        torchaudio.save(
            output_path / "test_wavs" / "{}-tar_wavsmatch.wav".format(idx),
            wavsmatch_tar.cpu(),
            tar_config["preprocess"]["sampling_rate"],
        )
        plot_and_save_mels(
            wavsdeg_src[0, ...].cpu().detach(),
            output_path / "test_mels" / "{}-src_melsdeg.png".format(idx),
            src_config,
        )
        plot_and_save_mels(
            wavsaux_tar[0, ...].cpu().detach(),
            output_path / "test_mels" / "{}-tar_melsaux.png".format(idx),
            tar_config,
        )
        if args.exist_src_aux:
            plot_and_save_mels(
                wavsdegbaseline_tar[0, ...].cpu().detach(),
                output_path / "test_mels" / "{}-tar_melsdegbaseline.png".format(idx),
                tar_config,
            )
            plot_and_save_mels(
                wavsdeg_tar[0, ...].cpu().detach(),
                output_path / "test_mels" / "{}-tar_melsdeg.png".format(idx),
                tar_config,
            )
        plot_and_save_mels(
            wavsmatch_tar[0, ...].cpu().detach(),
            output_path / "test_mels" / "{}-tar_melsmatch.png".format(idx),
            tar_config,
        )


if __name__ == "__main__":
    args = get_arg()
    chmatch_config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    output_path = pathlib.Path(chmatch_config["general"]["output_path"]) / args.run_name
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "test_wavs", exist_ok=True)
    os.makedirs(output_path / "test_mels", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args, chmatch_config, device)

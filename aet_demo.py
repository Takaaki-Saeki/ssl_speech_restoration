import pathlib
import yaml
import torch
import torchaudio
import numpy as np
from lightning_module import SSLDualLightningModule
import gradio as gr


def normalize_waveform(wav, sr, db=-3):
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav.unsqueeze(0),
        sr,
        [["norm", "{}".format(db)]],
    )
    return wav.squeeze(0)


def calc_spectrogram(wav, config):
    spec_module = torchaudio.transforms.MelSpectrogram(
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
    specs = spec_module(wav)
    log_spec = torch.log(
        torch.clamp_min(specs, config["preprocess"]["min_magnitude"])
        * config["preprocess"]["comp_factor"]
    ).to(torch.float32)
    return log_spec


def transfer(audio):
    wp_src = pathlib.Path("aet_sample/src.wav")
    wav_src, sr = torchaudio.load(wp_src)
    sr_inp, wav_tar = audio
    wav_tar = wav_tar / (np.max(np.abs(wav_tar)) * 1.1)
    wav_tar = torch.from_numpy(wav_tar.astype(np.float32))
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr_inp,
        new_freq=sr,
    )
    wav_tar = resampler(wav_tar)
    config_path = pathlib.Path("configs/test/melspec/ssl_tono.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    melspec_src = calc_spectrogram(normalize_waveform(wav_src.squeeze(0), sr), config)
    wav_tar = normalize_waveform(wav_tar.squeeze(0), sr)
    ckpt_path = pathlib.Path("aet_sample/tono_aet_melspec.ckpt")
    src_model = SSLDualLightningModule(config).load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
    )

    encoder_src = src_model.encoder
    channelfeats_src = src_model.channelfeats
    channel_src = src_model.channel

    _, enc_hidden_src = encoder_src(
        melspec_src.unsqueeze(0).unsqueeze(1).transpose(2, 3)
    )
    chfeats_src = channelfeats_src(enc_hidden_src)
    wav_transfer = channel_src(wav_tar.unsqueeze(0), chfeats_src)
    wav_transfer = wav_transfer.detach().numpy()[0, :]
    return sr, wav_transfer


if __name__ == "__main__":
    iface = gr.Interface(
        transfer,
        "audio",
        gr.outputs.Audio(type="numpy"),
        examples=[["aet_sample/tar.wav"]],
        title="Audio effect transfer demo",
        description="Add channel feature of Japanese old audio recording to any high-quality audio",
    )

    iface.launch()

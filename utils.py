import librosa.display
import matplotlib.pyplot as plt
import json
import torch
import torchaudio
import hifigan


def manual_logging(logger, item, idx, tag, global_step, data_type, config):

    if data_type == "audio":
        audio = item[idx, ...].detach().cpu().numpy()
        logger.add_audio(
            tag,
            audio,
            global_step,
            sample_rate=config["preprocess"]["sampling_rate"],
        )
    elif data_type == "image":
        image = item[idx, ...].detach().cpu().numpy()
        fig, ax = plt.subplots()
        _ = librosa.display.specshow(
            image,
            x_axis="time",
            y_axis="linear",
            sr=config["preprocess"]["sampling_rate"],
            hop_length=config["preprocess"]["frame_shift"],
            fmax=config["preprocess"]["sampling_rate"] // 2,
            ax=ax,
        )
        logger.add_figure(tag, fig, global_step)
    else:
        raise NotImplementedError(
            "Data type given to logger should be [audio] or [image]"
        )


def load_vocoder(config):
    with open(
        "hifigan/config_{}.json".format(config["general"]["feature_type"]), "r"
    ) as f:
        config_hifigan = hifigan.AttrDict(json.load(f))
    vocoder = hifigan.Generator(config_hifigan)
    vocoder.load_state_dict(torch.load(config["general"]["hifigan_path"])["generator"])
    vocoder.remove_weight_norm()
    for param in vocoder.parameters():
        param.requires_grad = False
    return vocoder


def get_conv_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def plot_and_save_mels(wav, save_path, config):
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
    spec = spec_module(wav.unsqueeze(0))
    log_spec = torch.log(
        torch.clamp_min(spec, config["preprocess"]["min_magnitude"])
        * config["preprocess"]["comp_factor"]
    )
    fig, ax = plt.subplots()
    _ = librosa.display.specshow(
        log_spec.squeeze(0).numpy(),
        x_axis="time",
        y_axis="linear",
        sr=config["preprocess"]["sampling_rate"],
        hop_length=config["preprocess"]["frame_shift"],
        fmax=config["preprocess"]["sampling_rate"] // 2,
        ax=ax,
        cmap="viridis",
    )
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def plot_and_save_mels_all(wavs, keys, save_path, config):
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
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    for i, key in enumerate(keys):
        wav = wavs[key][0, ...].cpu()
        spec = spec_module(wav.unsqueeze(0))
        log_spec = torch.log(
            torch.clamp_min(spec, config["preprocess"]["min_magnitude"])
            * config["preprocess"]["comp_factor"]
        )
        ax[i // 3, i % 3].set(title=key)
        _ = librosa.display.specshow(
            log_spec.squeeze(0).numpy(),
            x_axis="time",
            y_axis="linear",
            sr=config["preprocess"]["sampling_rate"],
            hop_length=config["preprocess"]["frame_shift"],
            fmax=config["preprocess"]["sampling_rate"] // 2,
            ax=ax[i // 3, i % 3],
            cmap="viridis",
        )
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


def configure_args(config, args):
    for key in ["stage", "corpus_type", "source_path", "aux_path", "preprocessed_path"]:
        if getattr(args, key) != None:
            config["general"][key] = str(getattr(args, key))

    for key in ["n_train", "n_val", "n_test"]:
        if getattr(args, key) != None:
            config["preprocess"][key] = getattr(args, key)

    for key in ["alpha", "beta", "learning_rate", "epoch"]:
        if getattr(args, key) != None:
            config["train"][key] = getattr(args, key)

    for key in ["load_pretrained", "early_stopping"]:
        config["train"][key] = getattr(args, key)

    if args.feature_loss_type != None:
        config["train"]["feature_loss"]["type"] = args.feature_loss_type

    for key in ["pretrained_path"]:
        if getattr(args, key) != None:
            config["train"][key] = str(getattr(args, key))

    return config, args

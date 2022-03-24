import argparse
import pathlib
import os
import tqdm
import soundfile as sf
import torch
import torchaudio
import numpy as np

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, type=pathlib.Path)
    parser.add_argument("--out_dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--corpus_type", required=True, type=str, choices=["single", "multi"]
    )
    parser.add_argument(
        "--deg_type",
        required=True,
        type=str,
        choices=["lowpass", "clipping", "mulaw", "overdrive"],
    )
    args = parser.parse_args()
    return args

def lowpass(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.corpus_type == "single":
        data_type = "single"
    else:
        data_type = "multi"

    os.makedirs(out_dir, exist_ok=True)

    if data_type == "single":
        wavslist = list(in_dir.glob("*.wav"))
    elif data_type == "multi":
        wavslist = list(in_dir.glob("*/*.wav"))
    else:
        raise NotImplementedError()

    for wp in tqdm.tqdm(wavslist):
        wav, sr = torchaudio.load(wp)
        if data_type == "multi":
            os.makedirs(out_dir / wp.parent.name, exist_ok=True)
        wav_processed = torchaudio.functional.lowpass_biquad(
            wav, sample_rate=sr, cutoff_freq=1000, Q=1.0
        )
        wav_out = wav_norm(wav_processed, sr)
        wav_out = wav_out.squeeze(0).numpy()
        if data_type == "single":
            sf.write(out_dir / wp.name, wav_out, sr)
        else:
            sf.write(out_dir / wp.parent.name / wp.name, wav_out, sr)

def clipping(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.corpus_type == "single":
        data_type = "single"
    else:
        data_type = "multi"

    os.makedirs(out_dir, exist_ok=True)

    if data_type == "single":
        wavslist = list(in_dir.glob("*.wav"))
    elif data_type == "multi":
        wavslist = list(in_dir.glob("*/*.wav"))
    else:
        raise NotImplementedError()

    eta = 0.25

    for wp in tqdm.tqdm(wavslist):
        wav, sr = sf.read(wp)
        if data_type == "multi":
            os.makedirs(out_dir / wp.parent.name, exist_ok=True)
        amp = eta * np.max(wav)
        wav_processed = np.maximum(np.minimum(wav, amp), -amp)
        wav_processed = torch.from_numpy(wav_processed.astype(np.float32)).unsqueeze(0)
        wav_out = wav_norm(wav_processed, sr)
        wav_out = wav_out.squeeze(0).numpy()
        if data_type == "single":
            sf.write(out_dir / wp.name, wav_out, sr)
        else:
            sf.write(out_dir / wp.parent.name / wp.name, wav_out, sr)

def mulaw(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.corpus_type == "single":
        data_type = "single"
    else:
        data_type = "multi"

    os.makedirs(out_dir, exist_ok=True)

    if data_type == "single":
        wavslist = list(in_dir.glob("*.wav"))
    elif data_type == "multi":
        wavslist = list(in_dir.glob("*/*.wav"))
    else:
        raise NotImplementedError()

    for wp in tqdm.tqdm(wavslist):
        wav, sr = sf.read(wp)
        if data_type == "multi":
            os.makedirs(out_dir / wp.parent.name, exist_ok=True)
        wav /= torch.max(torch.abs(torch.from_numpy(wav)))
        new_freq = 8000
        new_quantization = 128
        mulaw_encoder = torchaudio.transforms.MuLawEncoding(
            quantization_channels=new_quantization
        )
        wav_quantized = mulaw_encoder(wav) / new_quantization * 2.0 - 1.0
        downsampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=new_freq,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )
        upsampler = torchaudio.transforms.Resample(
            orig_freq=new_freq,
            new_freq=sr,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )
        wav_processed = upsampler(downsampler(wav_quantized))
        wav_out = wav_norm(wav_processed, sr)
        wav_out = wav_out.squeeze(0).numpy()
        if data_type == "single":
            sf.write(out_dir / wp.name, wav_out, sr)
        else:
            sf.write(out_dir / wp.parent.name / wp.name, wav_out, sr)

def overdrive(args):
    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.corpus_type == "single":
        data_type = "single"
    else:
        data_type = "multi"

    os.makedirs(out_dir, exist_ok=True)

    if data_type == "single":
        wavslist = list(in_dir.glob("*.wav"))
    elif data_type == "multi":
        wavslist = list(in_dir.glob("*/*.wav"))
    else:
        raise NotImplementedError()

    for wp in tqdm.tqdm(wavslist):
        wav, sr = sf.read(wp)
        if data_type == "multi":
            os.makedirs(out_dir / wp.parent.name, exist_ok=True)
        wav_processed = torchaudio.functional.overdrive(
            torch.from_numpy(wav.astype(np.float32)).unsqueeze(0), gain=40, colour=20
        )
        wav_out = wav_norm(wav_processed, sr)
        wav_out = wav_out.squeeze(0).numpy()
        if data_type == "single":
            sf.write(out_dir / wp.name, wav_out, sr)
        else:
            sf.write(out_dir / wp.parent.name / wp.name, wav_out, sr)

def wav_norm(wav_processed, sr):
    wav_out, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav_processed,
        sr,
        [["norm", "{}".format(-3)]],
    )
    return wav_out    

if __name__ == "__main__":
    args = get_arg()
    if args.deg_type == "lowpass":
        lowpass(args)
    elif args.deg_type == "clipping":
        clipping(args)
    elif args.deg_type == "mulaw":
        mulaw(args)
    elif args.deg_type == "overdrive":
        overdrive(args)
    else:
        raise NotImplementedError()

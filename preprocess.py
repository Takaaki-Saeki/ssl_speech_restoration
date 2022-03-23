import numpy as np
import os
import librosa
import tqdm
import pickle
import random
import argparse
import yaml
import pathlib


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=pathlib.Path)
    parser.add_argument("--corpus_type", default=None, type=str)
    parser.add_argument("--source_path", default=None, type=pathlib.Path)
    parser.add_argument("--source_path_task", default=None, type=pathlib.Path)
    parser.add_argument("--aux_path", default=None, type=pathlib.Path)
    parser.add_argument("--preprocessed_path", default=None, type=pathlib.Path)
    parser.add_argument("--n_train", default=None, type=int)
    parser.add_argument("--n_val", default=None, type=int)
    parser.add_argument("--n_test", default=None, type=int)
    return parser.parse_args()


def preprocess(config):

    # configs
    preprocessed_dir = pathlib.Path(config["general"]["preprocessed_path"])
    n_train = config["preprocess"]["n_train"]
    n_val = config["preprocess"]["n_val"]
    n_test = config["preprocess"]["n_test"]
    SR = config["preprocess"]["sampling_rate"]

    os.makedirs(preprocessed_dir, exist_ok=True)

    sourcepath = pathlib.Path(config["general"]["source_path"])

    if config["general"]["corpus_type"] == "single":
        fulllist = list(sourcepath.glob("*.wav"))
        random.seed(0)
        random.shuffle(fulllist)
        train_filelist = fulllist[:n_train]
        val_filelist = fulllist[n_train : n_train + n_val]
        test_filelist = fulllist[n_train + n_val : n_train + n_val + n_test]
        filelist = train_filelist + val_filelist + test_filelist
    elif config["general"]["corpus_type"] == "multi-seen":
        fulllist = list(sourcepath.glob("*/*.wav"))
        random.seed(0)
        random.shuffle(fulllist)
        train_filelist = fulllist[:n_train]
        val_filelist = fulllist[n_train : n_train + n_val]
        test_filelist = fulllist[n_train + n_val : n_train + n_val + n_test]
        filelist = train_filelist + val_filelist + test_filelist
    elif config["general"]["corpus_type"] == "multi-unseen":
        spk_list = list(set([x.parent for x in sourcepath.glob("*/*.wav")]))
        train_filelist = []
        val_filelist = []
        test_filelist = []
        random.seed(0)
        random.shuffle(spk_list)
        for i, spk in enumerate(spk_list):
            sourcespkpath = sourcepath / spk
            if i < n_train:
                train_filelist.extend(list(sourcespkpath.glob("*.wav")))
            elif i < n_train + n_val:
                val_filelist.extend(list(sourcespkpath.glob("*.wav")))
            elif i < n_train + n_val + n_test:
                test_filelist.extend(list(sourcespkpath.glob("*.wav")))
        filelist = train_filelist + val_filelist + test_filelist
    else:
        raise NotImplementedError(
            "corpus_type specified in config.yaml should be {single, multi-seen, multi-unseen}"
        )

    with open(preprocessed_dir / "train.txt", "w", encoding="utf-8") as f:
        for m in train_filelist:
            f.write(str(m) + "\n")
    with open(preprocessed_dir / "val.txt", "w", encoding="utf-8") as f:
        for m in val_filelist:
            f.write(str(m) + "\n")
    with open(preprocessed_dir / "test.txt", "w", encoding="utf-8") as f:
        for m in test_filelist:
            f.write(str(m) + "\n")

    for wp in tqdm.tqdm(filelist):

        if config["general"]["corpus_type"] == "single":
            basename = str(wp.stem)
        else:
            basename = str(wp.parent.name) + "-" + str(wp.stem)

        wav, _ = librosa.load(wp, sr=SR)
        wavsegs = []

        if config["general"]["aux_path"] != None:
            auxpath = pathlib.Path(config["general"]["aux_path"])
            if config["general"]["corpus_type"] == "single":
                wav_aux, _ = librosa.load(auxpath / wp.name, sr=SR)
            else:
                wav_aux, _ = librosa.load(auxpath / wp.parent.name / wp.name, sr=SR)
            wavauxsegs = []

        if config["general"]["aux_path"] == None:
            wavsegs.append(wav)
        else:
            min_seq_len = min(len(wav), len(wav_aux))
            wav = wav[:min_seq_len]
            wav_aux = wav_aux[:min_seq_len]
            wavsegs.append(wav)
            wavauxsegs.append(wav_aux)

        wavsegs = np.asarray(wavsegs).astype(np.float32)
        if config["general"]["aux_path"] != None:
            wavauxsegs = np.asarray(wavauxsegs).astype(np.float32)
        else:
            wavauxsegs = None

        d_preprocessed = {"wavs": wavsegs, "wavsaux": wavauxsegs}

        with open(preprocessed_dir / "{}.pickle".format(basename), "wb") as fw:
            pickle.dump(d_preprocessed, fw)


if __name__ == "__main__":
    args = get_arg()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    for key in ["corpus_type", "source_path", "aux_path", "preprocessed_path"]:
        if getattr(args, key) != None:
            config["general"][key] = str(getattr(args, key))
    for key in ["n_train", "n_val", "n_test"]:
        if getattr(args, key) != None:
            config["preprocess"][key] = getattr(args, key)

    print("Performing preprocessing ...")
    preprocess(config)

    if "dual" in config:
        if config["dual"]["enable"]:
            task_config = yaml.load(
                open(config["dual"]["config_path"], "r"), Loader=yaml.FullLoader
            )
            task_preprocessed_dir = (
                pathlib.Path(config["general"]["preprocessed_path"]).parent
                / pathlib.Path(task_config["general"]["preprocessed_path"]).name
            )
            task_config["general"]["preprocessed_path"] = task_preprocessed_dir
            if args.source_path_task != None:
                task_config["general"]["source_path"] = args.source_path_task
            print("Performing preprocessing for multi-task learning ...")
            preprocess(task_config)

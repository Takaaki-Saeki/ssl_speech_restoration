# SelfRemaster: Self-Supervised Speech Restoration

Official implementation of [SelfRemaster: Self-Supervised Speech Restoration with Analysis-by-Synthesis Approach Using Channel Modeling](https://arxiv.org/abs/2203.12937)

## Demo
- [Audio samples](https://takaaki-saeki.github.io/ssl_remaster_demo/)
- Audio effect transfer with [Gradio + HuggingFace Spaces 🤗](https://huggingface.co/spaces/saefro991/aet_demo)

## Setup
1. Clone this repository: `git clone https://github.com/Takaaki-Saeki/ssl_speech_restoration.git`.
2. CD into this repository: `cd ssl_speech_restoration`.
3. Install python packages and download some pretrained models: `./setup.sh`.

## Getting started
- If you use default Japanese corpora:
    - Download [JSUT Basic5000](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JVS Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
    - Downsample them to 22.05 kHz and place them under `data/` as `jsut_22k` and `jvs_22k`.
        - JSUT is a single-speaker dataset and requires the structure as `jsut_22k/*.wav`. Note that this is the ground-truth clean speech data which correspond to the simulated data and is not used for training. You may want to use `jsut_22k` only to compare the restored speech and ground-truth speech.
        - JVS parallel100 includes 100-speaker data and requires the structure as `jvs_22k/${spkr_name}/*.wav`. This is a clean speech dataset used for the backward learning of the dual-learning method. 
    - Place simulated low-quality data under `./data` as `jsut_22k-low`.
- Or you can use arbitrary datasets by modifying config files.

## Training

You can choose `MelSpec` or `SourFilter` models with `--config_path` option.  
As shown in the paper, `MelSpec` model is of higher-quality.

Firstly you need to split the data to train/val/test and dump them by the following command.
```shell
python preprocess.py --config_path configs/train/${feature}/ssl_jsut.yaml
```

To perform self-supervised learning with dual learning, run the following command.
```shell
python train.py \
    --config_path configs/train/${feature}/ssl_jsut.yaml \
    --stage ssl-dual \
    --run_name ssl_melspec_dual
```
For other options, refer to `train.py`.

## Speech restoration
To perform speech restoration of the test data, run the following command.
```shell
python eval.py \
    --config_path configs/test/${feature}/ssl_jsut.yaml \
    --ckpt_path ${path to checkpoint} \
    --stage ssl-dual \
    --run_name ssl_melspec_dual
```
For other options, see `eval.py`.

## Audio effect transfer
You can run a simple audio effect transfer demo using a model pretrained with real data.  
Run the following command.
```shell
python aet_demo.py
```

Or you can customize the dataset or model.  
You need to edit `audio_effect_transfer.yaml` and run the following command.
```shell
python aet.py \
    --config_path configs/test/melspec/audio_effect_transfer.yaml \
    --stage ssl-dual \
    --run_name aet_melspec_dual
```
For other options, see `aet.py`.


## Pretrained models
See [here](./pretrained_models.md).

## Reproducing results
You can generate simulated low-quality data as in the paper with the following command.
```shell
python simulated_data.py \
    --in_dir ${input_directory (e.g., path to jsut_22k)} \
    --output_dir ${output_directory (e.g., path to jsut_22k-low)} \
    --corpus_type ${single-speaker corpus or multi-speaker corpus} \
    --deg_type lowpass
```

Then download the pretrained model correspond to the deg_type and run the following command.
```shell
python eval.py \
    --config_path configs/train/${feature}/ssl_jsut.yaml \
    --ckpt_path ${path to checkpoint} \
    --stage ssl-dual \
    --run_name ssl_melspec_dual
```

## Citation
```bib
@article{saeki22selfremaster,
  title={{SelfRemaster}: {S}elf-Supervised Speech Restoration with Analysis-by-Synthesis Approach Using Channel Modeling},
  author={T. Saeki and S. Takamichi and T. Nakamura and N. Tanji and H. Saruwatari},
  journal={arXiv preprint arXiv:2203.12937},
  year={2022}
}
```

## Reference
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
- [VoiceFixer: Toward General Speech Restoration with Neural Vocoder](https://arxiv.org/abs/2109.13731)

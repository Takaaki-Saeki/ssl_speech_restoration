# Pretrained models

### Pretrained HiFi-GAN with SourceFilter features

HiFi-GAN-based synthethis modules to synthesize waveform from source-filter vocoder features trained on JVS or VCTK.  
Scripts for training are available in [another repo](https://github.com/Takaaki-Saeki/hifi-gan/tree/voc_feat).  
`hifigan_jvs_40d_600k` is used in the default configuration.

|Name|Feature|Dataset|Iteration|Link|
|------|---|---|---|---|
|hifigan_jvs_40d_600k|40-D Melcep. + F0 (WORLD)|JVS|600K|[Download](https://drive.google.com/file/d/1lkvtAJ3xTny5qmxyVcNPWQRB9MjdPlAY/view?usp=sharing)|
|hifigan_jvs_40d_1000k|40-D Melcep. + F0 (WORLD)|JVS|1000K|[Download](https://drive.google.com/file/d/1ZJbhWeAgs0RhoZ41puIKRFuioKyw0g8q/view?usp=sharing)|
|hifigan_vctk_40d_600k|40-D Melcep. + F0 (WORLD)|VCTK|600K|[Download](https://drive.google.com/file/d/1SnzZNt25eOCrrcMzF9KUjZ2kqVKE_cWf/view?usp=sharing)|
|hifigan_vctk-jvs_40d_400k|40-D Melcep. + F0 (WORLD)|JVS+VCTK|400K|[Download](https://drive.google.com/file/d/1I4HlZZZIXleKy7YtJP55MKpckwyvjZKm/view?usp=sharing)|
|hifigan_vctk-jvs_60d_400k|60-D Melcep. + F0 (WORLD)|JVS+VCTK|400K|[Download](https://drive.google.com/file/d/1kBJoTGgVSpGRkuEccZyJYiTLIim9kc48/view?usp=sharing)|

### SSL pretarined models for speech restoration

Speech restoration models trained on simulated data.

|Name|Dataset|Distortion|Feature|Link|
|------|---|---|---|---|
|jsut-bandlimited_melspec.ckpt|JSUT Baseic5000|Bandlimited|MelSpec|[Download](https://drive.google.com/file/d/1KwCEZ7pmfP__MjlE1sINnKs0Njw311m0/view?usp=sharing)|
|jsut-bandlimited_vocfeats.ckpt|JSUT Baseic5000|Bandlimited|SourceFilter|[Download](https://drive.google.com/file/d/1MB3FqxAHbDOWICib5tlEQ4DFM2e_7oJD/view?usp=sharing)|
|jsut-clip_melspec.ckpt|JSUT Baseic5000|Clipping|MelSpec|[Download](https://drive.google.com/file/d/19IkXv3rOwOeJ6TFNRp-x-cM4UWzNt6Ud/view?usp=sharing)|
|jsut-clip_vocfeats.ckpt|JSUT Baseic5000|Clipping|SourceFilter|[Download](https://drive.google.com/file/d/1_xfJqwJR-WhMSPZaTYE9xNqABQyFiu9m/view?usp=sharing)|
|jsut-qr_melspec.ckpt|JSUT Baseic5000|Quantized & Resampled|MelSpec|[Download](https://drive.google.com/file/d/1hn_q_hPROZlo_l89b0S2yPY2zUZk-RJf/view?usp=sharing)|
|jsut-qr_vocfeats.ckpt|JSUT Baseic5000|Quantized & Resampled|SourceFilter|[Download](https://drive.google.com/file/d/1_AdhP1KwdOKK_w6dZigiZ3yVe2vkCwyc/view?usp=sharing)|
|jsut-overdrive_melspec.ckpt|JSUT Baseic5000|Overdrive|MelSpec|[Download](https://drive.google.com/file/d/1I1Rhz8GwaUROPX8NOyqBKrAeDOSkaJTA/view?usp=sharing)|
|jsut-overdrive_vocfeats.ckpt|JSUT Baseic5000|Overdrive|SourceFilter|[Download](https://drive.google.com/file/d/1G_YjC8UZTTdDL93vCSQHu0lSiIF4_fmM/view?usp=sharing)|

### Supervisedly pretrained models

Supervisedly pretrained model to apply our method to low-resource settings.  
There are two type of the analysis module; `Normal` and `GST`.  
`Normal` is to extract restored speech features and channel features simultaneously in the analysis module.  
`GST` extracts channel features using a separated GST encoder.  
We use the `Normal` method in our paper because we have confirmed that the Normal method is of slightly higher quality in our preliminary experiments.  

|Name|Analysis module type|Feature|Dataset|Link|
|------|---|---|---|---|
|pretrain_melspec_normal.ckpt|Normal|MelSpec|JVS|[Download](https://drive.google.com/file/d/11bqYcyF0OqKogr4pDr7qeS7QysTbeOWd/view?usp=sharing)|
|pretrain_melspec_gst.ckpt|GST|MelSpec|JVS|[Download](https://drive.google.com/file/d/1vX9cTUnBFxjMfx_IP_RDtzEzi0_7z8ks/view?usp=sharing)|
|pretrain_vocfeats_normal.ckpt|Normal|SourceFilter|JVS|[Download](https://drive.google.com/file/d/1d2Nh9bbMEAW6gfy8PP0g3mFJImj8Tes9/view?usp=sharing)|
|pretrain_vocfeats_gst.ckpt|GST|SourceFilter|JVS|[Download](https://drive.google.com/file/d/1Qehs1sU0GSPX5VWqJs5tFaoxtzfPy11j/view?usp=sharing)|

### SSL pretarined models for audio effect transfer

The following model was trained on the real data described in the paper and is intended to be used for audio effect transfer.  
This operation enables to give effects to arbitrary speech data as if it were an old recording.  
Note that the following model uses `MelSpec` features.

|Name|Distortion|Link|
|------|---|---|
|tono.ckpt|Tono no mukashibanashi|[Download](https://drive.google.com/file/d/1xJzUNqwwf145YuSFQRZ4KjwxGtcL7rol/view?usp=sharing)|
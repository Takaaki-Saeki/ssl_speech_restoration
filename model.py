import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class EncoderModule(nn.Module):
    """
    Analysis module based on 2D conv U-Net
    Inspired by https://github.com/haoheliu/voicefixer

    Args:
        config (dict): config
        use_channel (bool): output channel feature or not
    """
    def __init__(self, config, use_channel=False):
        super().__init__()

        self.channels = 1
        self.use_channel = use_channel
        self.downsample_ratio = 2 ** 4

        self.down_block1 = DownBlockRes2D(
            in_channels=self.channels,
            out_channels=32,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.down_block2 = DownBlockRes2D(
            in_channels=32,
            out_channels=64,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.down_block3 = DownBlockRes2D(
            in_channels=64,
            out_channels=128,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.down_block4 = DownBlockRes2D(
            in_channels=128,
            out_channels=256,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.conv_block5 = ConvBlockRes2D(
            in_channels=256,
            out_channels=256,
            size=3,
            activation="relu",
            momentum=0.01,
        )
        self.up_block1 = UpBlockRes2D(
            in_channels=256,
            out_channels=256,
            stride=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.up_block2 = UpBlockRes2D(
            in_channels=256,
            out_channels=128,
            stride=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.up_block3 = UpBlockRes2D(
            in_channels=128,
            out_channels=64,
            stride=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.up_block4 = UpBlockRes2D(
            in_channels=64,
            out_channels=32,
            stride=(2, 2),
            activation="relu",
            momentum=0.01,
        )

        self.after_conv_block1 = ConvBlockRes2D(
            in_channels=32,
            out_channels=32,
            size=3,
            activation="relu",
            momentum=0.01,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        if config["general"]["feature_type"] == "melspec":
            out_dim = config["preprocess"]["n_mels"]
        elif config["general"]["feature_type"] == "vocfeats":
            out_dim = config["preprocess"]["cep_order"] + 1
        else:
            raise NotImplementedError()

        self.after_linear = nn.Linear(
            in_features=80,
            out_features=out_dim,
            bias=True,
        )

        if self.use_channel:
            self.conv_channel = ConvBlockRes2D(
                in_channels=256,
                out_channels=256,
                size=3,
                activation="relu",
                momentum=0.01,
            )

    def forward(self, x):
        """
        Forward
        
        Args:
            mel spectrogram: (batch, 1, time, freq)

        Return:
            speech feature (mel spectrogram or mel cepstrum): (batch, 1, time, freq)
            input of channel feature module (batch, 256, time, freq)
        """

        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0 : x.shape[-1] - 1]

        (x1_pool, x1) = self.down_block1(x)
        (x2_pool, x2) = self.down_block2(x1_pool)
        (x3_pool, x3) = self.down_block3(x2_pool)
        (x4_pool, x4) = self.down_block4(x3_pool)
        x_center = self.conv_block5(x4_pool)
        x5 = self.up_block1(x_center, x4)
        x6 = self.up_block2(x5, x3)
        x7 = self.up_block3(x6, x2)
        x8 = self.up_block4(x7, x1)
        x = self.after_conv_block1(x8)
        x = self.after_conv2(x)

        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        x = self.after_linear(x)

        if self.use_channel:
            x_channel = self.conv_channel(x4_pool)
            return x, x_channel
        else:
            return x


class ChannelModule(nn.Module):
    """
    Channel module based on 1D conv U-Net

    Args:
        config (dict): config
    """
    def __init__(self, config):
        super().__init__()

        self.channels = 1
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.down_block1 = DownBlockRes1D(
            in_channels=self.channels,
            out_channels=32,
            downsample=2,
            activation="relu",
            momentum=0.01,
        )
        self.down_block2 = DownBlockRes1D(
            in_channels=32,
            out_channels=64,
            downsample=2,
            activation="relu",
            momentum=0.01,
        )
        self.down_block3 = DownBlockRes1D(
            in_channels=64,
            out_channels=128,
            downsample=2,
            activation="relu",
            momentum=0.01,
        )
        self.down_block4 = DownBlockRes1D(
            in_channels=128,
            out_channels=256,
            downsample=2,
            activation="relu",
            momentum=0.01,
        )
        self.down_block5 = DownBlockRes1D(
            in_channels=256,
            out_channels=512,
            downsample=2,
            activation="relu",
            momentum=0.01,
        )
        self.conv_block6 = ConvBlockRes1D(
            in_channels=512,
            out_channels=384,
            size=3,
            activation="relu",
            momentum=0.01,
        )
        self.up_block1 = UpBlockRes1D(
            in_channels=512,
            out_channels=512,
            stride=2,
            activation="relu",
            momentum=0.01,
        )
        self.up_block2 = UpBlockRes1D(
            in_channels=512,
            out_channels=256,
            stride=2,
            activation="relu",
            momentum=0.01,
        )
        self.up_block3 = UpBlockRes1D(
            in_channels=256,
            out_channels=128,
            stride=2,
            activation="relu",
            momentum=0.01,
        )
        self.up_block4 = UpBlockRes1D(
            in_channels=128,
            out_channels=64,
            stride=2,
            activation="relu",
            momentum=0.01,
        )
        self.up_block5 = UpBlockRes1D(
            in_channels=64,
            out_channels=32,
            stride=2,
            activation="relu",
            momentum=0.01,
        )

        self.after_conv_block1 = ConvBlockRes1D(
            in_channels=32,
            out_channels=32,
            size=3,
            activation="relu",
            momentum=0.01,
        )

        self.after_conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x, h):
        """
        Forward

        Args:
            clean waveform: (batch, n_channel (1), time)
            channel feature: (batch, feature_dim)
        Outputs:
            degraded waveform: (batch, n_channel (1), time)
        """
        x = x.unsqueeze(1)

        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, pad_len))
        x = x[..., 0 : x.shape[-1] - 1]

        (x1_pool, x1) = self.down_block1(x)
        (x2_pool, x2) = self.down_block2(x1_pool)
        (x3_pool, x3) = self.down_block3(x2_pool)
        (x4_pool, x4) = self.down_block4(x3_pool)
        (x5_pool, x5) = self.down_block5(x4_pool)
        x_center = self.conv_block6(x5_pool)
        x_concat = torch.cat(
            (x_center, h.unsqueeze(2).expand(-1, -1, x_center.size(2))), dim=1
        )
        x6 = self.up_block1(x_concat, x5)
        x7 = self.up_block2(x6, x4)
        x8 = self.up_block3(x7, x3)
        x9 = self.up_block4(x8, x2)
        x10 = self.up_block5(x9, x1)
        x = self.after_conv_block1(x10)
        x = self.after_conv2(x)

        x = F.pad(x, pad=(0, 1))
        x = x[..., 0:origin_len]

        return x.squeeze(1)


class ChannelFeatureModule(nn.Module):
    """
    Channel feature module based on 2D convolution layers

    Args:
        config (dict): config
    """
    def __init__(self, config):
        super().__init__()
        self.conv_blocks_in = ConvBlockRes2D(
            in_channels=256,
            out_channels=512,
            size=3,
            activation="relu",
            momentum=0.01,
        )
        self.down_block1 = DownBlockRes2D(
            in_channels=512,
            out_channels=256,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.down_block2 = DownBlockRes2D(
            in_channels=256,
            out_channels=256,
            downsample=(2, 2),
            activation="relu",
            momentum=0.01,
        )
        self.conv_block_out = ConvBlockRes2D(
            in_channels=256,
            out_channels=128,
            size=3,
            activation="relu",
            momentum=0.01,
        )
        self.avgpool2d = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward

        Args:
            output of analysis module: (batch, 256, time, freq)
        
        Return:
            channel feature: (batch, feature_dim)
        """
        x = self.conv_blocks_in(x)
        x, _ = self.down_block1(x)
        x, _ = self.down_block2(x)
        x = self.conv_block_out(x)
        x = self.avgpool2d(x)
        x = x.squeeze(3).squeeze(2)
        return x


class ConvBlockRes2D(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super().__init__()

        self.activation = activation
        if type(size) == type((3, 4)):
            pad = size[0] // 2
            size = size[0]
        else:
            pad = size // 2
            size = size

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(pad, pad),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(pad, pad),
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class ConvBlockRes1D(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super().__init__()

        self.activation = activation
        pad = size // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=1,
            padding=pad,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(in_channels, momentum=momentum)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=1,
            padding=pad,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels, momentum=momentum)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class DownBlockRes2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super().__init__()
        size = 3

        self.conv_block1 = ConvBlockRes2D(
            in_channels, out_channels, size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )
        self.avg_pool2d = torch.nn.AvgPool2d(downsample)

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = self.avg_pool2d(encoder)
        return encoder_pool, encoder


class DownBlockRes1D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super().__init__()
        size = 3

        self.conv_block1 = ConvBlockRes1D(
            in_channels, out_channels, size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )
        self.avg_pool1d = torch.nn.AvgPool1d(downsample)

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = self.avg_pool1d(encoder)
        return encoder_pool, encoder


class UpBlockRes2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super().__init__()
        size = 3
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, size),
            stride=stride,
            padding=(0, 0),
            output_padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes2D(
            out_channels * 2, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes2D(
            out_channels, out_channels, size, activation, momentum
        )

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution."""
        if both:
            x = x[:, :, 0:-1, 0:-1]
        else:
            x = x[:, :, 0:-1, :]
        return x

    def forward(self, input_tensor, concat_tensor, both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class UpBlockRes1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super().__init__()
        size = 3
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=stride,
            padding=0,
            output_padding=0,
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv_block2 = ConvBlockRes1D(
            out_channels * 2, out_channels, size, activation, momentum
        )
        self.conv_block3 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block4 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )
        self.conv_block5 = ConvBlockRes1D(
            out_channels, out_channels, size, activation, momentum
        )

    def prune(self, x):
        """Prune the shape of x after transpose convolution."""
        print(x.shape)
        x = x[:, 0:-1, :]
        print(x.shape)
        return x

    def forward(self, input_tensor, concat_tensor):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        # x = self.prune(x)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi scale spectral loss
    https://openreview.net/forum?id=B1x1ma4tDr

    Args:
        config (dict): config
    """
    def __init__(self, config):
        super().__init__()
        try:
            self.use_linear = config["train"]["multi_scale_loss"]["use_linear"]
            self.gamma = config["train"]["multi_scale_loss"]["gamma"]
        except KeyError:
            self.use_linear = False

        self.fft_sizes = [2048, 512, 256, 128, 64]
        self.spectrograms = []
        for fftsize in self.fft_sizes:
            self.spectrograms.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=fftsize, hop_length=fftsize // 4, power=2
                )
            )
        self.spectrograms = nn.ModuleList(self.spectrograms)
        self.criteria = nn.L1Loss()
        self.eps = 1e-10

    def forward(self, wav_out, wav_target):
        """
        Forward

        Args:
            wav_out: output of channel module (batch, time)
            wav_target: input degraded waveform (batch, time)
        
        Return:
            loss
        """
        loss = 0.0
        length = min(wav_out.size(1), wav_target.size(1))
        for spectrogram in self.spectrograms:
            S_out = spectrogram(wav_out[..., :length])
            S_target = spectrogram(wav_target[..., :length])
            log_S_out = torch.log(S_out + self.eps)
            log_S_target = torch.log(S_target + self.eps)
            if self.use_linear:
                loss += self.criteria(S_out, S_target) + self.gamma * self.criteria(
                    log_S_out, log_S_target
                )
            else:
                loss += self.criteria(log_S_out, log_S_target)
        return loss


class ReferenceEncoder(nn.Module):
    def __init__(
        self, idim=80, ref_enc_filters=[32, 32, 64, 64, 128, 128], ref_dim=128
    ):
        super().__init__()
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)]
        )

        out_channels = self.calculate_channels(idim, 3, 2, 1, K)

        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=ref_dim,
            batch_first=True,
        )
        self.n_mel_channels = idim

    def forward(self, inputs):

        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()

        _, out = self.gru(out)

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    def __init__(self, ref_dim=128, num_heads=4, token_num=10, token_dim=128):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, token_dim // num_heads))
        d_q = ref_dim
        d_k = token_dim // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_dim, num_heads=num_heads
        )
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)
        return style_embed


class MultiHeadAttention(nn.Module):
    """
    Multi head attention
    https://github.com/KinglittleQ/GST-Tacotron

    """
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False
        )
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False
        )

    def forward(self, query, key):
        """
        Forward

        Args:
            query: (batch, T_q, query_dim)
            key: (batch, T_k, key_dim)
        
        Return:
            out: (N, T_q, num_units)
        """
        querys = self.W_query(query)  # [N, T_q, num_units]

        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(
            torch.split(querys, split_size, dim=2), dim=0
        )  # [h, N, T_q, num_units/h]
        keys = torch.stack(
            torch.split(keys, split_size, dim=2), dim=0
        )  # [h, N, T_k, num_units/h]
        values = torch.stack(
            torch.split(values, split_size, dim=2), dim=0
        )  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(
            0
        )  # [N, T_q, num_units]

        return out


class GSTModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_post = ReferenceEncoder(
            idim=config["preprocess"]["n_mels"],
            ref_dim=256,
        )
        self.stl = STL(ref_dim=256, num_heads=8, token_num=10, token_dim=128)

    def forward(self, inputs):
        acoustic_embed = self.encoder_post(inputs)
        style_embed = self.stl(acoustic_embed)
        return style_embed.squeeze(1)

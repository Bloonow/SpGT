import copy
import os
import sys
import numpy as np
import torch
from torch import Tensor
from SpGT.dataset.matio import write_matfile

from SpGT.module.layer import DownInterpolate2D, FeedForward, SimpleAttention, SpectralConv2D, UpInterpolate2D


class DownScaler2D(torch.nn.Module):
    # downsample using interpolate
    def __init__(self, in_dim, out_dim, interp_size, droprate, activation) -> None:
        super().__init__()
        self.downsample_layer = DownInterpolate2D(
            in_dim=in_dim, out_dim=out_dim, interp_size=interp_size,
            droprate=droprate, activation=activation
        )

    def forward(self, X: Tensor):
        """
        Input  : [N, r_in, r_in, in_dim]
        Output : [N, r_out, r_out, out_dim]
        """
        X = X.permute(0, 3, 1, 2)
        X = self.downsample_layer(X)
        X = X.permute(0, 2, 3, 1)
        return X


class UpScaler2D(torch.nn.Module):
    # upsample using interpolate
    def __init__(self, in_dim, out_dim, interp_size, droprate, activation) -> None:
        super().__init__()
        self.upsample_layer = UpInterpolate2D(
            in_dim=in_dim, out_dim=out_dim, interp_size=interp_size,
            droprate=droprate, activation=activation
        )

    def forward(self, X: Tensor):
        """
        Input  : [N, r_in, r_in, in_dim]
        Output : [N, r_out, r_out, out_dim]
        """
        X = X.permute(0, 3, 1, 2)
        X = self.upsample_layer(X)
        X = X.permute(0, 2, 3, 1)
        return X


class SimpleEncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
        init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
    ) -> None:
        super().__init__()
        self.attention_layer = SimpleAttention(
            d_model=d_model, n_head=n_head, d_pos=d_pos, norm_eps=norm_eps,
            init_xavier_uniform_gain=init_xavier_uniform_gain, init_diagonal_weight=init_diagonal_weight,
            init_symmetric=init_symmetric, droprate=droprate
        )
        self.ffn_layer = FeedForward(
            in_dim=d_model, out_dim=d_model, ffn_dim=d_encoder_ffn_hidden, droprate=droprate
        )
        self.dropout_layer = torch.nn.Dropout(p=droprate)

    def forward(self, X: Tensor, pos: Tensor = None):
        """
        Inputs:
            X   : [N, seqlen, d_model]
            pod : [N, seqlen, d_pos]
        Output  : [N, seqlen, d_model]
        """
        res = X
        X = self.attention_layer(X, X, X, pos)
        X = self.dropout_layer(X) + res
        res = X
        X = self.ffn_layer(X)
        X = self.dropout_layer(X) + res
        return X


class PointwiseRegressor(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim, n_regressor_layer,
        use_spacial_fc, spacial_fc_dim, droprate, activation
    ) -> None:
        super().__init__()
        self.use_spacial_fc = use_spacial_fc

        if use_spacial_fc:
            self.spacial_fc_layer = torch.nn.Linear(in_dim + spacial_fc_dim, hidden_dim)
        ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU(),
            torch.nn.Dropout(p=droprate)
        )
        self.ffn_layers = torch.nn.ModuleList([copy.deepcopy(ffn) for _ in range(n_regressor_layer)])
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, X: Tensor, grid: Tensor = None):
        """
        Input  : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        if self.use_spacial_fc:
            X = torch.concat([X, grid], dim=-1)
            X = self.spacial_fc_layer(X)
        for layer in self.ffn_layers:
            X = layer(X)
        X = self.out_layer(X)
        return X


class SpectralRegressor(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim, freq_dim, n_regressor_layer, modes,
        use_spacial_fc, spacial_fc_dim, droprate, activation
    ) -> None:
        super().__init__()
        self.use_spacial_fc = use_spacial_fc
        if self.use_spacial_fc:
            self.spacial_fc_layer = torch.nn.Linear(in_dim + spacial_fc_dim, hidden_dim)
        self.spectral_conv_layers = torch.nn.ModuleList([SpectralConv2D(
            in_dim=hidden_dim, out_dim=freq_dim, modes=modes, droprate=droprate, activation=activation
        ),])
        for _ in range(n_regressor_layer - 1):
            self.spectral_conv_layers.append(SpectralConv2D(
                in_dim=freq_dim, out_dim=freq_dim, modes=modes, droprate=droprate, activation=activation
            ))
        ffn_dim = 2 * spacial_fc_dim * freq_dim
        self.ffn_layer = torch.nn.Sequential(
            torch.nn.Linear(freq_dim, ffn_dim),
            torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, out_dim)
        )

    def forward(self, X: Tensor, grid: Tensor = None):
        """
        Input  : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        if self.use_spacial_fc:
            X = torch.concat([X, grid], dim=-1)
            X = self.spacial_fc_layer(X)
        for layer in self.spectral_conv_layers:
            X = layer(X)
        X = self.ffn_layer(X)
        return X


class GalerkinTransformer2D(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.d_hidden = cfg['d_hidden']
        self.boundary_condition = cfg['boundary_condition']
        self.target_normalizer = cfg['target_normalizer']

        # 在 Encoder 前后分别进行两次插值，分别为两次降采样插值与两次升采样插值
        # Data -> downsample x2 -> Encoders -> upsample x2 -> Decoders -> Output
        self.downscaler = DownScaler2D(
            cfg['d_node'], cfg['d_hidden'], cfg['downscaler_size'], cfg['downscaler_droprate'], cfg['downscaler_activation']
        )
        self.upscaler = UpScaler2D(
            cfg['d_hidden'], cfg['d_hidden'], cfg['upscaler_size'], cfg['upscaler_droprate'], cfg['upscaler_activation']
        )
        encoder = SimpleEncoderLayer(
            cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
            cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
        )
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])])
        self.regressor_layer = SpectralRegressor(
            cfg['d_hidden'], cfg['d_target'], cfg['d_frequency'], cfg['d_frequency'], cfg['n_regressor_layer'], cfg['d_fourier_mode'],
            cfg['use_spacial_fc'], cfg['d_pos'], cfg['decoder_droprate'], cfg['regressor_activation']
        )
        self.dropout_layer = torch.nn.Dropout(p=cfg['gt_droprate'])

    def forward(self, X: Tensor, pos: Tensor, grid: Tensor):
        """
        Input:
            X    : [N, r, r, 1]
            grid : [N, r, r, 2]
            pos  : [N, r_attn, r_attn, 2]
        Output   : [N, r, r, 1]
        """
        N, r_attn, r_attn, d_pos = pos.size()
        X = self.downscaler(X)
        X = self.dropout_layer(X)
        X = X.view(N, r_attn * r_attn, self.d_hidden)
        pos = pos.view(N, r_attn * r_attn, d_pos)
        for encoder in self.encoder_layers:
            X = encoder(X, pos=pos)
        X = X.view(N, r_attn, r_attn, self.d_hidden)
        X = self.upscaler(X)
        X = self.dropout_layer(X)
        X = self.regressor_layer(X, grid=grid)
        if self.target_normalizer:
            X = self.target_normalizer.inverse_transform(X)
        if self.boundary_condition == 'dirichlet':
            X = X[:, 1:-1, 1:-1].contiguous()
            X = torch.nn.functional.pad(X, (0, 0, 1, 1, 1, 1), 'constant', 0)
        return X

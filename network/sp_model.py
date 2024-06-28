import torch
from torch import Tensor

from SpGT.network.layer import FeedForward
from SpGT.network.model import DownScaler2D, UpScaler2D
from SpGT.network.sp_layer import Sp_EncoderLayer, Sp_SpectralConv2D


class Sp_EncoderMixer(torch.nn.Module):
    def __init__(
        self, num_layer,
        dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Sp_EncoderLayer(
                dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
                ffn_dim_hidden, ffn_droprate, ffn_activation
            ) for _ in range(num_layer)
        ])

    def forward(self, X: Tensor, position: Tensor):
        """
        X        : [N, r_attn, r_attn, dim_hidden]   --> [N, seqlen, dim_hidden]
        position : [N, r_attn, r_attn, dim_position] --> [N, seqlen, dim_position]
        Output   : [N, r_attn, r_attn, dim_hidden]   <-- [N, seqlen, dim_hidden]
        """
        N, r_attn, r_attn, dim_hidden = X.size()
        dim_position = position.size(dim=-1)
        X = X.view(N, r_attn * r_attn, dim_hidden)
        position = position.view(N, r_attn * r_attn, dim_position)
        for layer in self.layers:
            X = layer(X, position)
        X = X.view(N, r_attn, r_attn, dim_hidden)
        return X


class Sp_DecoderRegressor(torch.nn.Module):
    def __init__(
        self, num_layer,
        in_dim, out_dim, dim_position, dim_spatial_hidden, mode, droprate, activation
    ) -> None:
        super().__init__()
        self.spatial_fc_layer = torch.nn.Linear(in_dim + dim_position, dim_spatial_hidden)
        self.layers = torch.nn.ModuleList([
            Sp_SpectralConv2D(dim_spatial_hidden, dim_spatial_hidden, mode, droprate, activation)
            for _ in range(num_layer)
        ])
        dim_ffn_hidden = 2 * dim_position * dim_spatial_hidden
        self.ffn_layer = FeedForward(dim_spatial_hidden, out_dim, dim_ffn_hidden, droprate, activation)

    def forward(self, X: Tensor, grid: Tensor = None):
        """
        X      : [N, r, r, dim_hidden]
        grid   : [N, r, r, dim_position]
        Output : [N, r, r, dim_target]
        """
        X = torch.concat([X, grid], dim=-1)
        X = self.spatial_fc_layer(X)
        for layer in self.layers:
            X = layer(X)
        X = self.ffn_layer(X)
        return X


class Sp_GalerkinTransformer2D(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.droprate = cfg['drop_module']
        self.boundary_condition = cfg['boundary_condition']
        self.normalizer = cfg['target_normalizer']
        # 在 Encoder 前后分别进行两次插值，分别为两次降采样插值与两次升采样插值
        # Data -> downsample x2 -> Encoders -> upsample x2 -> Decoders -> Output
        self.down_scaler = DownScaler2D(
            cfg['resolution_size'], cfg['dim_node'], cfg['dim_hidden'], cfg['drop_downscaler'], cfg['acti_downscaler']
        )
        self.encoder_mixer = Sp_EncoderMixer(
            cfg['num_encoder_layer'], cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
            cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
            cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
        )
        self.up_scaler = UpScaler2D(
            cfg['resolution_size'], cfg['dim_hidden'], cfg['dim_hidden'], cfg['drop_upscaler'], cfg['acti_upscaler']
        )
        self.decoder_regressor = Sp_DecoderRegressor(
            cfg['num_decoder_layer'], cfg['dim_hidden'], cfg['dim_target'], cfg['dim_position'],
            cfg['dim_spatial_hidden'], cfg['num_frequence_mode'], cfg['drop_decoder_layer'], cfg['acti_decoder_layer']
        )

    def forward(self, X: Tensor, position: Tensor, grid: Tensor):
        """
        X        : [N, r, r, dim_node]
        position : [N, r_attn, r_attn, dim_position]
        grid     : [N, r, r, dim_position]
        """
        # X : [N, r, r, dim_node]
        X = self.down_scaler(X)
        X = torch.nn.functional.dropout(X, p=self.droprate)
        # X : [N, r_attn, r_attn, dim_hidden]
        X = self.encoder_mixer(X, position)
        # X : [N, r_attn, r_attn, dim_hidden]
        X = self.up_scaler(X)
        X = torch.nn.functional.dropout(X, p=self.droprate)
        # X : [N, r, r, dim_hidden]
        X = self.decoder_regressor(X, grid)
        # X : [N, r, r, dim_target]
        # 后续逆归一化与边界条件
        X = self.normalizer.inverse_transform(X)
        if self.boundary_condition == 'dirichlet':
            X = X[:, 1:-1, 1:-1].contiguous()
            X = torch.nn.functional.pad(X, (0, 0, 1, 1, 1, 1), 'constant', 0)
        return X

import torch
from torch import Tensor

from SpGT.network.layer import ConvResBlock2D, EncoderLayer, FeedForward, SpectralConv2D


class DownScaler2D(torch.nn.Module):
    def __init__(
        self, resolution_size, in_dim, out_dim, droprate, activation
    ) -> None:
        super().__init__()
        self.resolution_size = resolution_size
        out_dim1 = out_dim // 3
        out_dim2 = out_dim // 3
        out_dim3 = int(out_dim - out_dim1 - out_dim2)
        kernel_size, padding = 3, 1
        self.conv0 = ConvResBlock2D(in_dim, out_dim, kernel_size, padding, droprate, activation)
        self.conv1 = ConvResBlock2D(out_dim, out_dim1, kernel_size, padding, droprate, activation)
        self.conv2 = ConvResBlock2D(out_dim1, out_dim2, kernel_size, padding, droprate, activation)
        self.conv3 = ConvResBlock2D(out_dim2, out_dim3, kernel_size, padding, droprate, activation)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X: Tensor):
        """
        X      : [N, r, r, in_dim]             -->  [N, in_dim, H_in, W_in]
        Output : [N, r_attn, r_attn, out_dim]  <--  [N, out_dim, H_out, W_out]
        """
        X = torch.permute(X, [0, 3, 1, 2])
        X = self.conv0(X)
        X = torch.nn.functional.interpolate(X, size=self.resolution_size[1], mode='bilinear', align_corners=True)
        X = self.activation_layer(X)
        X1 = self.conv1(X)
        X2 = self.conv2(X1)
        X3 = self.conv3(X2)
        X = torch.concat([X1, X2, X3], dim=1)  # 在 channel 维度上拼接
        X = torch.nn.functional.interpolate(X, size=self.resolution_size[2], mode='bilinear', align_corners=True)
        X = self.activation_layer(X)
        X = torch.permute(X, [0, 2, 3, 1])
        return X


class UpScaler2D(torch.nn.Module):
    def __init__(
        self, resolution_size, in_dim, out_dim, droprate, activation
    ) -> None:
        super().__init__()
        self.resolution_size = resolution_size
        self.droprate = droprate
        kernel_size, padding = 3, 1
        self.conv = ConvResBlock2D(in_dim, out_dim, kernel_size, padding, droprate, activation)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X: Tensor):
        """
        X      : [N, r_attn, r_attn, in_dim]  -->  [N, in_dim, H_in, W_in]
        Output : [N, r, r, out_dim]           <--  [N, out_dim, H_out, W_out]
        """
        X = torch.permute(X, [0, 3, 1, 2])
        X = torch.nn.functional.interpolate(X, size=self.resolution_size[3], mode='bilinear', align_corners=True)
        X = self.conv(X)
        X = torch.nn.functional.dropout(X, p=self.droprate)
        X = self.activation_layer(X)
        X = torch.nn.functional.interpolate(X, size=self.resolution_size[4], mode='bilinear', align_corners=True)
        X = torch.permute(X, [0, 2, 3, 1])
        return X


class EncoderMixer(torch.nn.Module):
    def __init__(
        self, num_layer,
        dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            EncoderLayer(
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


class DecoderRegressor(torch.nn.Module):
    def __init__(
        self, num_layer,
        in_dim, out_dim, dim_position, dim_spatial_hidden, mode, droprate, activation
    ) -> None:
        super().__init__()
        self.spatial_fc_layer = torch.nn.Linear(in_dim + dim_position, dim_spatial_hidden)
        self.layers = torch.nn.ModuleList([
            SpectralConv2D(dim_spatial_hidden, dim_spatial_hidden, mode, droprate, activation)
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


class GalerkinTransformer2D(torch.nn.Module):
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
        self.encoder_mixer = EncoderMixer(
            cfg['num_encoder_layer'], cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
            cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
            cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
        )
        self.up_scaler = UpScaler2D(
            cfg['resolution_size'], cfg['dim_hidden'], cfg['dim_hidden'], cfg['drop_upscaler'], cfg['acti_upscaler']
        )
        self.decoder_regressor = DecoderRegressor(
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

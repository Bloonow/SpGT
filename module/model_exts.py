import copy
import torch
from SpGT.module.layer_exts import SimpleAttention_Exts, SpectralConv2D_Exts
from SpGT.module.model import GalerkinTransformer2D, PointwiseRegressor, SimpleEncoderLayer, SpectralRegressor


class SimpleEncoderLayer_Exts(SimpleEncoderLayer):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
        init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
    ) -> None:
        super().__init__(
            d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
            init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
        )
        self.attention_layer = SimpleAttention_Exts(
            d_model=d_model, n_head=n_head, d_pos=d_pos, norm_eps=norm_eps,
            init_xavier_uniform_gain=init_xavier_uniform_gain, init_diagonal_weight=init_diagonal_weight,
            init_symmetric=init_symmetric, droprate=droprate
        )


class SpectralRegressor_Exts(SpectralRegressor):
    def __init__(
        self, in_dim, out_dim, hidden_dim, freq_dim, n_regressor_layer, modes,
        use_spacial_fc, spacial_fc_dim, droprate, activation
    ) -> None:
        super().__init__(
            in_dim, out_dim, hidden_dim, freq_dim, n_regressor_layer, modes,
            use_spacial_fc, spacial_fc_dim, droprate, activation
        )
        self.spectral_conv_layers = torch.nn.ModuleList([SpectralConv2D_Exts(
            in_dim=hidden_dim, out_dim=freq_dim, modes=modes, droprate=droprate, activation=activation
        ),])
        for _ in range(n_regressor_layer - 1):
            self.spectral_conv_layers.append(SpectralConv2D_Exts(
                in_dim=freq_dim, out_dim=freq_dim, modes=modes, droprate=droprate, activation=activation
            ))


class GalerkinTransformer2D_Exts(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # 在 Encoder 前后分别进行两次插值，分别为两次降采样插值与两次升采样插值
        # Data -> downsample x2 -> Encoders -> upsample x2 -> Decoders -> Output
        encoder = SimpleEncoderLayer_Exts(
            cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
            cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
        )
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])])
        self.regressor_layer = SpectralRegressor_Exts(
            cfg['d_hidden'], cfg['d_target'], cfg['d_frequency'], cfg['d_frequency'], cfg['n_regressor_layer'], cfg['d_fourier_mode'],
            cfg['use_spacial_fc'], cfg['d_pos'], cfg['decoder_droprate'], cfg['regressor_activation']
        )

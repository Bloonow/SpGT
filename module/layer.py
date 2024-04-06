from functools import partial
from math import sqrt
import torch
from torch import Tensor


class ConvResBlock2D(torch.nn.Module):
    def __init__(
        self, in_chans, out_chans, kernel_size=3, stride=1, padding=1, dilation=1,
        use_bias=False, use_basic_block=False, use_residual=False, droprate=0.0, activation='silu'
    ) -> None:
        super().__init__()
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=use_bias
            ),
            torch.nn.Dropout(p=droprate)
        )
        self.use_basic_block = use_basic_block
        if use_basic_block:
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=out_chans, out_chans=out_chans, kernel_size=kernel_size,
                    padding=padding, bias=use_bias
                ),
                torch.nn.Dropout(p=droprate)
            )
        self.use_residual = use_residual
        if use_residual:
            if in_chans != out_chans:
                self.residual_layer = torch.nn.Linear(in_features=in_chans, out_features=out_chans)
            else:
                self.residual_layer = torch.nn.Identity()
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X: Tensor):
        """
        Input  : [N, in_chans, H_in, W_in]
        Output : [N, out_chans, H_out, W_out]
        """
        if self.use_residual:
            res = torch.permute(self.residual_layer(X.permute(0, 2, 3, 1)), dims=[0, 3, 1, 2])
        X = self.conv0(X)
        if self.use_basic_block:
            X = self.conv1(X)
        if self.use_residual:
            return self.activation_layer(X + res)
        else:
            return self.activation_layer(X)


class DownInterpolate2D(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, interp_size=None, kernel_size=3, stride=1, padding=1, dilation=1,
        use_residual=False, droprate=0.0, activation='silu'
    ) -> None:
        super().__init__()
        self.interp_size = interp_size
        self.use_residual = use_residual
        conv_out_dim0 = out_dim
        conv_out_dim1 = out_dim // 3
        conv_out_dim2 = out_dim // 3
        conv_out_dim3 = int(out_dim - conv_out_dim1 - conv_out_dim2)
        padding0 = padding
        padding1 = max(padding // 2, 1)
        padding2 = max(padding // 4, 1)
        self.conv0 = ConvResBlock2D(
            in_chans=in_dim, out_chans=conv_out_dim0, kernel_size=kernel_size,
            stride=stride, padding=padding0, dilation=dilation,
            use_residual=use_residual, droprate=droprate, activation=activation
        )
        self.conv1 = ConvResBlock2D(
            in_chans=conv_out_dim0, out_chans=conv_out_dim1, kernel_size=kernel_size,
            stride=stride, padding=padding1, dilation=dilation,
            use_residual=use_residual, droprate=droprate, activation=activation
        )
        self.conv2 = ConvResBlock2D(
            in_chans=conv_out_dim1, out_chans=conv_out_dim2, kernel_size=kernel_size,
            stride=stride, padding=padding2, dilation=dilation,
            use_residual=use_residual, droprate=droprate, activation=activation
        )
        self.conv3 = ConvResBlock2D(
            in_chans=conv_out_dim2, out_chans=conv_out_dim3, kernel_size=kernel_size,
            stride=stride, dilation=dilation,
            use_residual=use_residual, droprate=droprate, activation=activation
        )
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X):
        """
        Input  : [N, in_chans, H_in, W_in]
        Output : [N, out_chans, H_out, W_out]
        """
        X = self.conv0(X)
        X = torch.nn.functional.interpolate(
            X, size=self.interp_size[0], mode='bilinear', align_corners=True
        )
        res = self.activation_layer(X)
        X1 = self.conv1(res)
        X2 = self.conv2(X1)
        X3 = self.conv3(X2)
        X = torch.concat([X1, X2, X3], dim=1)  # 在 channel 维度上拼接
        if self.use_residual:
            X = X + res
        X = torch.nn.functional.interpolate(
            X, size=self.interp_size[1], mode='bilinear', align_corners=True
        )
        X = self.activation_layer(X)
        return X


class UpInterpolate2D(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, interp_size=None, kernel_size=3, stride=1, padding=1, dilation=1,
        use_conv_block=True, use_residual=False, droprate=0.0, activation='silu'
    ) -> None:
        super().__init__()
        self.interp_size = interp_size
        self.use_conv_block = use_conv_block
        if use_conv_block:
            self.conv = torch.nn.Sequential(
                ConvResBlock2D(
                    in_chans=in_dim, out_chans=out_dim, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    use_residual=use_residual, droprate=droprate, activation=activation
                )
            )
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X):
        """
        Input  : [N, in_chans, H_in, W_in]
        Output : [N, out_chans, H_out, W_out]
        """
        X = torch.nn.functional.interpolate(
            X, size=self.interp_size[0], mode='bilinear', align_corners=True
        )
        if self.use_conv_block:
            X = self.conv(X)
        X = torch.nn.functional.interpolate(
            X, size=self.interp_size[1], mode='bilinear', align_corners=True
        )
        return X


class SimpleAttention(torch.nn.Module):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps=1.e-5,
        init_xavier_uniform_gain=1.e-2, init_diagonal_weight=1.e-2, init_symmetric=False, droprate=0.0
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0,  'Error d_model % n_head != 0'
        assert d_model / n_head == 32, 'Error d_model / n_head != 32'
        d_k = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_pos = d_pos
        self.norm_eps = norm_eps
        self.droprate = droprate
        self.init_xavier_uniform_gain = init_xavier_uniform_gain
        self.init_diagonal_weight = init_diagonal_weight
        self.init_symmetric = init_symmetric

        self.weight_Q = torch.nn.Parameter(torch.empty([d_model, d_model]))
        self.weight_K = torch.nn.Parameter(torch.empty([d_model, d_model]))
        self.weight_V = torch.nn.Parameter(torch.empty([d_model, d_model]))
        self.bias_Q = torch.nn.Parameter(torch.empty([d_model]))
        self.bias_K = torch.nn.Parameter(torch.empty([d_model]))
        self.bias_V = torch.nn.Parameter(torch.empty([d_model]))
        self.lnw_K = torch.nn.Parameter(torch.empty([n_head, d_k]))
        self.lnb_K = torch.nn.Parameter(torch.empty([n_head, d_k]))
        self.lnw_V = torch.nn.Parameter(torch.empty([n_head, d_k]))
        self.lnb_V = torch.nn.Parameter(torch.empty([n_head, d_k]))
        if d_pos > 0:
            self.fc_layer = torch.nn.Linear(d_model + d_pos * n_head, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for weight in [self.weight_Q, self.weight_K, self.weight_V]:
            torch.nn.init.xavier_uniform_(weight, gain=self.init_xavier_uniform_gain)
            if self.init_diagonal_weight > 0:
                weight.data += self.init_diagonal_weight * torch.diag(torch.ones(weight.size(-1)))
            if self.init_symmetric:
                weight.data += weight.data.T
        for bias in [self.bias_Q, self.bias_K, self.bias_V]:
            torch.nn.init.constant_(bias, 0)

        for lnw, lnb in zip([self.lnw_K, self.lnw_V], [self.lnb_K, self.lnb_V]):
            torch.nn.init.ones_(lnw)
            torch.nn.init.zeros_(lnb)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos: Tensor = None):
        """
        Inputs:
            Q,K,V : [N, seqlen, d_model]
            pos   : [N, seqlen, d_pos]
        Output    : [N, seqlen, d_model]
        """
        N, seqlen, d_model = Q.size()
        Q = self.projpos_orig(Q, self.weight_Q, self.bias_Q, pos, self.n_head, self.d_k, self.d_pos)
        K = self.projpos_lnorm_orig(
            K, self.weight_K, self.bias_K, self.lnw_K, self.lnb_K, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        V = self.projpos_lnorm_orig(
            V, self.weight_V, self.bias_V, self.lnw_V, self.lnb_V, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        Attn = self.galattn_orig(Q, K, V, droprate=self.droprate)  # [N, n_head, seqlen, d_pos + d_k]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.n_head * (self.d_pos + self.d_k))
        output = self.fc_layer(Attn)
        return output

    @staticmethod
    def projpos_orig(
        input: Tensor, weight: Tensor, bias: Tensor, pos: Tensor,
        n_head, d_k, d_pos
    ) -> Tensor:
        """
        input  : [batch, seqlen, d_model]
        weight : [d_model, d_model]
        bias   : [d_model]
        pos    : [batch, seqlen, d_pos]
        """
        batch, seqlen, d_model = input.size()
        X = torch.nn.functional.linear(input, weight, bias)
        X = X.view(batch, seqlen, n_head, d_k).transpose(1, 2)
        # X = [batch, n_head, seqlen, d_k]
        p = pos.unsqueeze(dim=1).repeat([1, n_head, 1, 1])
        X = torch.concat([p, X], dim=-1)
        return X

    @staticmethod
    def projpos_lnorm_orig(
        input: Tensor, weight: Tensor, bias: Tensor, lnw: Tensor, lnb: Tensor, pos: Tensor,
        norm_eps, n_head, d_k, d_pos
    ) -> Tensor:
        """
        input   : [batch, seqlen, d_model]
        weight  : [d_model, d_model]
        bias    : [d_model]
        pos     : [batch, seqlen, d_pos]
        lnw,lnb : [n_head, d_k]
        """
        batch, seqlen, d_model = input.size()
        X = torch.nn.functional.linear(input, weight, bias)
        X = X.view(batch, seqlen, n_head, d_k).transpose(1, 2)
        # X = [batch, n_head, seqlen, d_k]
        X = torch.stack([
            torch.nn.functional.layer_norm(inp, [d_k,], lnorm_weight, lnorm_bias, norm_eps)
            for lnorm_weight, lnorm_bias, inp in zip(
                [lnw[i, ...] for i in range(n_head)],
                [lnb[i, ...] for i in range(n_head)],
                [X[:, i, ...] for i in range(n_head)]
            )
        ], dim=1)
        # X = [batch, n_head, seqlen, d_k]
        p = pos.unsqueeze(dim=1).repeat([1, n_head, 1, 1])
        X = torch.concat([p, X], dim=-1)
        return X

    @staticmethod
    def galattn_orig(
        Q: Tensor, K: Tensor, V: Tensor, droprate=0.0
    ) -> Tensor:
        """
        Q,K,V : [batch, n_head, seqlen, d_posk]
        """
        seqlen = Q.size(dim=-2)
        A = torch.div(torch.matmul(K.transpose(-2, -1), V), seqlen)
        if 0.0 < droprate < 1.0:
            A = torch.nn.functional.dropout(input=A, p=droprate)
        A = torch.matmul(Q, A)
        return A


class FeedForward(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, ffn_dim,
        use_batch_norm=False, droprate=0.0, activation='relu'
    ) -> None:
        super().__init__()
        self.linear_layer0 = torch.nn.Linear(in_dim, ffn_dim)
        self.linear_layer1 = torch.nn.Linear(ffn_dim, out_dim)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=droprate)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(num_features=ffn_dim)

    def forward(self, X: Tensor):
        """
        Input  : [N, seqlen, dim]
        Output : [N, seqlen, dim]
        """
        X = self.linear_layer0(X)
        X = self.activation_layer(X)
        X = self.dropout_layer(X)
        if self.use_batch_norm:
            X = X.permute(0, 2, 1)
            X = self.batch_norm_layer(X)
            X = X.permute(0, 2, 1)
        X = self.linear_layer1(X)
        return X


class SpectralConv2D(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, modes, droprate=0.0, activation='silu', norm_type='ortho'
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.norm_type = norm_type
        self.residual_layer = torch.nn.Linear(in_dim, out_dim)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=droprate)
        # 未将 Real 与 Imag 分开时的傅里叶参数
        self.fourier_weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty([in_dim, out_dim, modes, modes, 2])) for _ in range(2)
        ])
        for param in self.fourier_weight:
            torch.nn.init.xavier_normal_(param, gain=1. / (in_dim * out_dim) * sqrt(in_dim + out_dim))

    @staticmethod
    def complex_matmul_2D(feat, weight):
        # (batch, in_dim, x, y), (in_dim, out_dim, x, y) -> (batch, out_dim, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(feat[..., 0], weight[..., 0]) - op(feat[..., 1], weight[..., 1]),
            op(feat[..., 1], weight[..., 0]) + op(feat[..., 0], weight[..., 1])
        ], dim=-1)

    def forward(self, X: Tensor):
        """
        Input  : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        N, r, r, _ = X.size()
        modes = self.modes

        res = self.residual_layer(X)
        X = self.dropout_layer(X)

        X = X.permute(0, 3, 1, 2).contiguous()
        X_ft = torch.fft.rfft2(X, s=(r, r), norm=self.norm_type)
        X_ft = torch.stack([X_ft.real, X_ft.imag], dim=-1)
        out_ft = torch.zeros(N, self.out_dim, r, r // 2 + 1, 2, device=X.device)
        out_ft[:, :, :modes, :modes] = self.complex_matmul_2D(X_ft[:, :, :modes, :modes], self.fourier_weight[0])
        out_ft[:, :, -modes:, :modes] = self.complex_matmul_2D(X_ft[:, :, -modes:, :modes], self.fourier_weight[1])
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
        X = torch.fft.irfft2(out_ft, s=(r, r), norm=self.norm_type)
        X = X.permute(0, 2, 3, 1).contiguous()

        X = self.activation_layer(X + res)
        return X

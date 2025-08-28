from functools import partial
from math import sqrt
import torch
from torch import Tensor


class ConvResBlock2D(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size, padding, droprate, activation
    ) -> None:
        super().__init__()
        self.droprate = droprate
        self.conv_layer = torch.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X: Tensor):
        """
        X      : [N, in_dim, H_in, W_in]
        Output : [N, out_dim, H_out, W_out]
        """
        X = self.conv_layer(X)
        X = torch.nn.functional.dropout(X, p=self.droprate)
        X = self.activation_layer(X)
        return X


class GalerkinAttention(torch.nn.Module):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric
    ) -> None:
        super().__init__()
        assert dim_hidden % 128 == 0 and dim_hidden // num_head == 32
        dim_head = dim_hidden // num_head  # 32
        self.dim_hidden = dim_hidden
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_position = dim_position
        self.norm_eps = norm_eps
        self.attn_droprate = attn_droprate
        self.fc_droprate = fc_droprate
        self.weight_Q = torch.nn.Parameter(torch.empty([dim_hidden, dim_hidden]))
        self.weight_K = torch.nn.Parameter(torch.empty([dim_hidden, dim_hidden]))
        self.weight_V = torch.nn.Parameter(torch.empty([dim_hidden, dim_hidden]))
        self.bias_Q = torch.nn.Parameter(torch.empty([dim_hidden]))
        self.bias_K = torch.nn.Parameter(torch.empty([dim_hidden]))
        self.bias_V = torch.nn.Parameter(torch.empty([dim_hidden]))
        self.ln_weight_K = torch.nn.Parameter(torch.empty([num_head, dim_head]))
        self.ln_weight_V = torch.nn.Parameter(torch.empty([num_head, dim_head]))
        self.ln_bias_K = torch.nn.Parameter(torch.empty([num_head, dim_head]))
        self.ln_bias_V = torch.nn.Parameter(torch.empty([num_head, dim_head]))
        self.fc_layer = torch.nn.Linear(dim_hidden + dim_position * num_head, dim_hidden)
        self._reset_parameters(xavier, diagonal, symmetric)

    def _reset_parameters(self, xavier, diagonal, symmetric):
        for weight in [self.weight_Q, self.weight_K, self.weight_V]:
            torch.nn.init.xavier_uniform_(weight, gain=xavier)
            if diagonal > 0:
                weight.data += diagonal * torch.diag(torch.ones(weight.size(-1)))
            if symmetric:
                weight.data += weight.data.T
        for bias in [self.bias_Q, self.bias_K, self.bias_V]:
            torch.nn.init.constant_(bias, 0)
        for lnw, lnb in zip([self.ln_weight_K, self.ln_weight_V], [self.ln_bias_K, self.ln_bias_V]):
            torch.nn.init.ones_(lnw)
            torch.nn.init.zeros_(lnb)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, position: Tensor):
        """
        Inputs:
            Q,K,V    : [N, seqlen, dim_hidden]
            position : [N, seqlen, dim_position]
        Output       : [N, seqlen, dim_hidden]
        """
        N, seqlen, dim_hidden = Q.size()
        Q = self.multihead_projection_with_position(
            Q, self.weight_Q, self.bias_Q, position,
            self.num_head, self.dim_head, self.dim_position
        )
        K = self.multihead_projection_layernorm_with_position(
            K, self.weight_K, self.bias_K, self.ln_weight_K, self.ln_bias_K, position,
            self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        V = self.multihead_projection_layernorm_with_position(
            V, self.weight_V, self.bias_V, self.ln_weight_V, self.ln_bias_V, position,
            self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        X = self.multihead_galerkin_attention(Q, K, V, droprate=self.attn_droprate)
        # X : [N, num_head, seqlen, dim_position + dim_head]
        X = X.transpose(1, 2).contiguous().view(N, seqlen, self.num_head * (self.dim_position + self.dim_head))
        X = self.fc_layer(X)
        X = torch.nn.functional.dropout(X, p=self.fc_droprate)
        # X : [N, seqlen, dim_hidden]
        return X

    @staticmethod
    def multihead_projection_with_position(
        input: Tensor, weight: Tensor, bias: Tensor, position: Tensor,
        num_head, dim_head, dim_position
    ) -> Tensor:
        """
        input    : [N, seqlen, dim_hidden]
        weight   : [dim_hidden, dim_hidden]
        bias     : [dim_hidden]
        position : [N, seqlen, dim_position]
        """
        N, seqlen, dim_hidden = input.size()
        X = torch.nn.functional.linear(input, weight, bias)
        X = X.view(N, seqlen, num_head, dim_head).transpose(1, 2)
        # X : [N, num_head, seqlen, dim_head]
        pos = position.unsqueeze(dim=1).repeat([1, num_head, 1, 1])
        X = torch.concat([pos, X], dim=-1)
        return X

    @staticmethod
    def multihead_projection_layernorm_with_position(
        input: Tensor, weight: Tensor, bias: Tensor, ln_weight: Tensor, ln_bias: Tensor, position: Tensor,
        norm_eps, num_head, dim_head, dim_position
    ) -> Tensor:
        """
        input             : [N, seqlen, dim_hidden]
        weight            : [dim_hidden, dim_hidden]
        bias              : [dim_hidden]
        ln_weight,ln_bias : [num_head, dim_head]
        position          : [N, seqlen, dim_position]
        """
        N, seqlen, dim_hidden = input.size()
        X = torch.nn.functional.linear(input, weight, bias)
        X = X.view(N, seqlen, num_head, dim_head).transpose(1, 2)
        # X : [N, num_head, seqlen, dim_head]
        X = torch.stack([
            torch.nn.functional.layer_norm(head, [dim_head,], head_ln_weight, head_ln_bias, norm_eps)
            for head, head_ln_weight, head_ln_bias in zip(
                [X[:, i, ...] for i in range(num_head)],
                [ln_weight[i, ...] for i in range(num_head)],
                [ln_bias[i, ...] for i in range(num_head)]
            )
        ], dim=1)
        # X : [N, num_head, seqlen, dim_head]
        pos = position.unsqueeze(dim=1).repeat([1, num_head, 1, 1])
        X = torch.concat([pos, X], dim=-1)
        return X

    @staticmethod
    def multihead_galerkin_attention(Q: Tensor, K: Tensor, V: Tensor, droprate) -> Tensor:
        """
        Q,K,V : [N, num_head, seqlen, dim_position + dim_head]
        """
        seqlen = Q.size(dim=-2)
        X = torch.div(torch.matmul(K.transpose(-2, -1), V), seqlen)
        X = torch.nn.functional.dropout(X, p=droprate)
        X = torch.matmul(Q, X)
        return X


class FeedForward(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, dim_hidden, droprate, activation
    ) -> None:
        super().__init__()
        self.droprate = droprate
        self.layer1 = torch.nn.Linear(in_dim, dim_hidden)
        self.layer2 = torch.nn.Linear(dim_hidden, out_dim)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()

    def forward(self, X: Tensor):
        """
        X      : [N, seqlen, in_dim]
        Output : [N, seqlen, out_dim]
        """
        X = self.layer1(X)
        X = self.activation_layer(X)
        X = self.layer2(X)
        X = torch.nn.functional.dropout(X, p=self.droprate)
        return X


class EncoderLayer(torch.nn.Module):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__()
        self.attention_layer = GalerkinAttention(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric
        )
        self.ffn_layer = FeedForward(
            dim_hidden, dim_hidden, ffn_dim_hidden, ffn_droprate, ffn_activation
        )

    def forward(self, X: Tensor, position: Tensor):
        """
        X      : [N, seqlen, dim_hidden]
        pos    : [N, seqlen, dim_position]
        Output : [N, seqlen, dim_hidden]
        """
        res = X
        X = res + self.attention_layer(X, X, X, position)
        res = X
        X = res + self.ffn_layer(X)
        return X


class SpectralConv2D(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, mode, droprate, activation
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode
        self.droprate = droprate
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()
        self.residual_layer = torch.nn.Linear(in_dim, out_dim)
        self.fourier_weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty([in_dim, out_dim, mode, mode, 2])) for _ in range(2)
        ])
        for para in self.fourier_weight:
            torch.nn.init.xavier_normal_(para, gain=1.0 / (in_dim * out_dim) * sqrt(in_dim + out_dim))

    def forward(self, X: Tensor):
        """
        X      : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        N, r, r, in_dim = X.size()
        res = self.residual_layer(X)  # [N, r, r, out_dim]
        X = torch.nn.functional.dropout(X, p=self.droprate)

        # X : [N, r, r, in_dim]
        X = torch.permute(X, [0, 3, 1, 2])
        # X : [N, in_dim, r, r]
        X_ft = torch.fft.rfft2(X, s=(r, r), norm='ortho')
        X_ft = torch.stack([X_ft.real, X_ft.imag], dim=-1)
        # X_ft : [N, in_dim, r, r // 2 + 1, 2]
        m = self.mode
        Out_ft = torch.zeros(size=[N, self.out_dim, r, r // 2 + 1, 2], device=X_ft.device)
        Out_ft[:, :, :m, :m] = self.complex_matmul_2D(X_ft[:, :, :m, :m], self.fourier_weight[0])
        Out_ft[:, :, -m:, :m] = self.complex_matmul_2D(X_ft[:, :, -m:, :m], self.fourier_weight[1])
        Out_ft = torch.complex(Out_ft[..., 0], Out_ft[..., 1])
        # Out_ft : [N, self.out_dim, r, r // 2 + 1]
        X = torch.fft.irfft2(Out_ft, s=(r, r), norm='ortho')
        # X : [N, out_dim, r, r]
        X = torch.permute(X, [0, 2, 3, 1])
        # X : [N, r, r, out_dim]

        X = self.activation_layer(X + res)
        return X

    @staticmethod
    def complex_matmul_2D(X, weight):
        """
        X      : [N, in_dim, H, W, 2]
        weight : [in_dim, out_dim, H, W, 2]
        Output : [N, out_dim, H, W, 2]
        """
        op = partial(torch.einsum, "nihw,iohw->nohw")
        X = torch.stack([
            op(X[..., 0], weight[..., 0]) - op(X[..., 1], weight[..., 1]),
            op(X[..., 1], weight[..., 0]) + op(X[..., 0], weight[..., 1])
        ], dim=-1)
        return X

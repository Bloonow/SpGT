import math
import torch
from torch import Tensor

from SpGT.extension.bind.galerkin_attention import multihead_projection_with_position_rrc_cuda
from SpGT.extension.bind.galerkin_attention import multihead_projection_layernorm_with_position_rrc_cuda
from SpGT.extension.bind.galerkin_attention import multihead_galerkin_attention_cccr_cuda
from SpGT.extension.bind.datamove import batched_transpose, transpose_gather_dual, transpose_scatter_dual
from SpGT.network.layer import FeedForward


class Sp_GalerkinAttention(torch.nn.Module):
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
        Q = multihead_projection_with_position_rrc_cuda(
            Q, self.weight_Q, self.bias_Q, position,
            self.num_head, self.dim_head, self.dim_position
        )
        K = multihead_projection_layernorm_with_position_rrc_cuda(
            K, self.weight_K, self.bias_K, self.ln_weight_K, self.ln_bias_K, position,
            self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        V = multihead_projection_layernorm_with_position_rrc_cuda(
            V, self.weight_V, self.bias_V, self.ln_weight_V, self.ln_bias_V, position,
            self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        X = multihead_galerkin_attention_cccr_cuda(Q, K, V)
        # X : [N, num_head, seqlen, dim_position + dim_head]
        X = X.transpose(1, 2).contiguous().view(N, seqlen, self.num_head * (self.dim_position + self.dim_head))
        X = self.fc_layer(X)
        X = torch.nn.functional.dropout(X, p=self.fc_droprate)
        # X : [N, seqlen, dim_hidden]
        return X


class Sp_EncoderLayer(torch.nn.Module):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__()
        self.attention_layer = Sp_GalerkinAttention(
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


class Sp_SpectralConv2D(torch.nn.Module):
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
        # 分布式时 NCCL 后端不支持 torch.complex64 数据类型
        self.fourier_weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty([mode, mode, in_dim, out_dim, 2])) for _ in range(2)
        ])
        for param in self.fourier_weight:
            torch.nn.init.xavier_normal_(param, gain=1.0 / (in_dim * out_dim) * math.sqrt(in_dim + out_dim))

    def forward(self, X: Tensor):
        """
        Input  : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        N, r, r, in_dim = X.size()
        out_dim, m = self.out_dim, self.mode
        res = self.residual_layer(X)  # [N, r, r, out_dim]
        X = torch.nn.functional.dropout(X, p=self.droprate)

        # X : [N, r, r, in_dim]
        X = batched_transpose(X, [N,], [r, r], [in_dim,])
        # X : [N, in_dim, r, r]
        X_ft = torch.fft.rfft2(X, s=(r, r), norm='ortho')
        X_ft_0, X_ft_1 = transpose_gather_dual(
            X_ft, [N, in_dim], r, r // 2 + 1, m, m, 0, 0, r-m, 0
        )
        # Out_ft_0 = X_ft_0 @ self.fourier_weight[0]
        # Out_ft_1 = X_ft_1 @ self.fourier_weight[1]
        Out_ft_0 = X_ft_0 @ torch.view_as_complex(self.fourier_weight[0])
        Out_ft_1 = X_ft_1 @ torch.view_as_complex(self.fourier_weight[1])
        Out_ft = transpose_scatter_dual(
            Out_ft_0, Out_ft_1, [N, out_dim], r, r // 2 + 1, m, m, 0, 0, r-m, 0
        )
        # Out_ft : [N, out_dim, r, r // 2 + 1]
        X = torch.fft.irfft2(Out_ft, s=(r, r), norm='ortho')
        # X : [N, out_dim, r, r]
        X = batched_transpose(X, [N,], [out_dim,], [r, r])
        # X : [N, r, r, out_dim]

        X = self.activation_layer(X + res)
        return X

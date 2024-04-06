import math
import torch
from torch import Tensor

from SpGT.exts.bind.galerkin import galattn_cccr_cuda, projpos_lnorm_rrc_cuda, projpos_rrc_cuda
from SpGT.exts.bind.dmove import batched_transpose_2D, gather_transpose_dual, scatter_transpose_dual
from SpGT.module.layer import SimpleAttention


class SimpleAttention_Exts(SimpleAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos: Tensor = None):
        """
        Inputs:
            Q,K,V : [N, seqlen, d_model]
            pod   : [N, seqlen, d_pos]
        Output    : [N, seqlen, d_model]
        """
        N, seqlen, d_model = Q.size()
        Q = projpos_rrc_cuda(Q, self.weight_Q, self.bias_Q, pos, self.n_head, self.d_k, self.d_pos)
        K = projpos_lnorm_rrc_cuda(
            K, self.weight_K, self.bias_K, self.lnw_K, self.lnb_K, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        V = projpos_lnorm_rrc_cuda(
            V, self.weight_V, self.bias_V, self.lnw_V, self.lnb_V, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        Attn = galattn_cccr_cuda(Q, K, V)  # self.droprate  # [N, n_head, seqlen, d_pos + d_k]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.n_head * (self.d_pos + self.d_k))
        output = self.fc_layer(Attn)
        return output


class SpectralConv2D_Exts(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, modes, droprate=0.0, activation='silu', norm_type='ortho'
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.norm_type = norm_type

        self.residual_layer = torch.nn.Linear(in_dim, out_dim)
        self.dropout_layer = torch.nn.Dropout(p=droprate)
        self.activation_layer = torch.nn.SiLU() if activation == 'silu' else torch.nn.ReLU()
        self.fourier_weight = torch.nn.ParameterList([
            # 直接采用 torch.complex64 数据类型，在分布式时 NCCL 后端不支持
            torch.nn.Parameter(torch.empty([modes, modes, in_dim, out_dim, 2])) for _ in range(2)
        ])
        for param in self.fourier_weight:
            torch.nn.init.xavier_normal_(param, gain=1. / (in_dim * out_dim) * math.sqrt(in_dim + out_dim))

    def forward(self, X: Tensor):
        """
        Input  : [N, r, r, in_dim]
        Output : [N, r, r, out_dim]
        """
        N, r, r, _ = X.size()
        in_dim, out_dim, modes = self.in_dim, self.out_dim, self.modes

        res = self.residual_layer(X)
        X = self.dropout_layer(X)

        X = batched_transpose_2D(X, [r, r], [in_dim,], [N,])       # [N, in_dim, r, r]
        x_ft = torch.fft.rfft2(X, s=(r, r), norm=self.norm_type)   # [N, in_dim, r, r // 2 + 1]
        x_ft_positive, x_ft_negative = gather_transpose_dual(
            x_ft, [N, in_dim], r // 2 + 1, r, modes, modes, N1_offset1=0, N1_offset2=r-modes
        )
        # out_ft_positive = x_ft_positive @ self.fourier_weight[0]
        # out_ft_negative = x_ft_negative @ self.fourier_weight[1]
        out_ft_positive = x_ft_positive @ torch.view_as_complex(self.fourier_weight[0])
        out_ft_negative = x_ft_negative @ torch.view_as_complex(self.fourier_weight[1])
        out_ft = scatter_transpose_dual(
            out_ft_positive, out_ft_negative, [N, out_dim], r // 2 + 1, r, modes, modes, N1_offset1=0, N1_offset2=r-modes
        )
        # out_ft = [N, out_dim, r, r // 2 + 1]
        X = torch.fft.irfft2(out_ft, s=(r, r), norm=self.norm_type)  # [N, out_dim, r, r]
        X = batched_transpose_2D(X, [out_dim,], [r, r], [N,])        # [N, r, r, out_dim]

        X = self.activation_layer(X + res)
        return X

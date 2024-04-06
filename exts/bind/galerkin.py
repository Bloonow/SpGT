import os
import torch
from torch import Tensor
from typing import Any
import torch.utils.cpp_extension

from SpGT.common.path import EXTS_PATH
galerkin = torch.utils.cpp_extension.load(
    'galerkin', os.path.join(EXTS_PATH, 'bind', 'galerkin.cu'), verbose=True
)


class MH32_ProjPos_rrc_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, weight: Tensor, bias: Tensor, pos: Tensor, n_head, d_k, d_pos):
        batch, seqlen, d_model = input.size()
        out = galerkin.mh32_projpos_rrc_forward(input, weight, bias, pos, seqlen, d_model, n_head, d_k, d_pos, batch)
        ctx.save_for_backward(input, weight)
        ctx.shape_kwargs = [n_head, d_k, d_pos]
        # output = [batch, n_head, d_pos + d_k, seqlen]
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        input, weight, = ctx.saved_tensors
        n_head, d_k, d_pos = ctx.shape_kwargs
        batch, seqlen, d_model = input.size()
        grad_input, grad_weight, grad_bias = galerkin.mh32_projpos_rrc_backward(
            grad_out, input, weight, seqlen, d_model, n_head, d_k, d_pos, batch
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class MH32_ProjPos_Lnorm_rrc_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: Tensor, weight: Tensor, bias: Tensor, lnw: Tensor, lnb: Tensor, pos: Tensor,
        norm_eps: float, n_head, d_k, d_pos
    ):
        batch, seqlen, d_model = input.size()
        # 此处输出的并不是纯粹的 hat = (x - mu) * invsigma
        # 而是层归一化之后的结果 hat_ln = (x - mu) * invsigma * lnw + lnb
        hat_ln, invsigma = galerkin.mh32_projpos_lnorm_rrc_forward(
            input, weight, bias, lnw, lnb, norm_eps, pos, seqlen, d_model, n_head, d_k, d_pos, batch
        )
        ctx.save_for_backward(input, weight, hat_ln, invsigma, lnw, lnb)
        ctx.shape_kwargs = [n_head, d_k, d_pos]
        # hat_ln = [batch, n_head, d_pos + d_k, seqlen]
        return hat_ln

    @staticmethod
    def backward(ctx: Any, grad_ln: Tensor):
        input, weight, hat_ln, invsigma, lnw, lnb, = ctx.saved_tensors
        n_head, d_k, d_pos = ctx.shape_kwargs
        batch, seqlen, d_model = input.size()
        grad_input, grad_weight, grad_bias, grad_lnw, grad_lnb = galerkin.mh32_projpos_lnorm_rrc_backward(
            grad_ln, input, weight, hat_ln, invsigma, lnw, lnb, seqlen, d_model, n_head, d_k, d_pos, batch
        )
        return grad_input, grad_weight, grad_bias, grad_lnw, grad_lnb, None, None, None, None, None


class GalAttn_cccr_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, Q: Tensor, K: Tensor, V: Tensor):
        batch, n_head, d_posk, seqlen = Q.size()
        Attn = galerkin.mh32_galattn_cccr_forward(Q, K, V, batch, n_head, d_posk, seqlen)
        ctx.save_for_backward(Q, K, V)
        # Attn = [batch, n_head, seqlen, d_posk]
        return Attn

    @staticmethod
    def backward(ctx: Any, grad_Attn):
        Q, K, V = ctx.saved_tensors
        batch, n_head, d_posk, seqlen = Q.size()
        grad_Q, grad_K, grad_V = galerkin.mh32_galattn_cccr_backward(grad_Attn, Q, K, V, batch, n_head, d_posk, seqlen)
        return grad_Q, grad_K, grad_V


def projpos_rrc_cuda(
    input: Tensor, weight: Tensor, bias: Tensor, pos: Tensor, n_head, d_k, d_pos
) -> Tensor:
    return MH32_ProjPos_rrc_Function.apply(input, weight, bias, pos, n_head, d_k, d_pos)


def projpos_lnorm_rrc_cuda(
    input: Tensor, weight: Tensor, bias: Tensor, lnw: Tensor, lnb: Tensor, pos: Tensor, norm_eps: float, n_head, d_k, d_pos
) -> Tensor:
    return MH32_ProjPos_Lnorm_rrc_Function.apply(input, weight, bias, lnw, lnb, pos, norm_eps, n_head, d_k, d_pos)


def galattn_cccr_cuda(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    return GalAttn_cccr_Function.apply(Q, K, V)


def batched_skinny_gemm(A: Tensor, B: Tensor) -> Tensor:
    return galerkin.batched_skinny_gemm(A, B, 1.)

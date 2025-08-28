import os
import torch
from torch import Tensor
from typing import Any
import torch.utils.cpp_extension
from SpGT.common.path import EXTENSION_PATH

gexts = torch.utils.cpp_extension.load(
    'galerkin_attention', os.path.join(EXTENSION_PATH, 'bind', 'galerkin_attention.cu'), verbose=True
)

class Multihead_Porjection_with_Position_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: Tensor, weight: Tensor, bias: Tensor, position: Tensor, num_head, dim_head, dim_position
    ):
        batch, seqlen, d_model = input.size()
        out = gexts.multihead_projection_with_position_rrc_forward(
            input, weight, bias, position, seqlen, d_model, num_head, dim_head, dim_position, batch
        )
        ctx.save_for_backward(input, weight)
        ctx.shape_kwargs = [num_head, dim_head, dim_position]
        # output = [batch, num_head, dim_position + dim_head, seqlen]
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        input, weight, = ctx.saved_tensors
        num_head, dim_head, dim_position = ctx.shape_kwargs
        batch, seqlen, d_model = input.size()
        grad_input, grad_weight, grad_bias = gexts.multihead_projection_with_position_rrc_backward(
            grad_out, input, weight, seqlen, d_model, num_head, dim_head, dim_position, batch
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class Multihead_Porjection_Layernorm_with_Position_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: Tensor, weight: Tensor, bias: Tensor, lnw: Tensor, lnb: Tensor, position: Tensor,
        norm_eps: float, num_head, dim_head, dim_position
    ):
        batch, seqlen, d_model = input.size()
        # 此处输出的并不是纯粹的 hat = (x - mu) * invsigma
        # 而是层归一化之后的结果 hat_ln = (x - mu) * invsigma * lnw + lnb
        hat_ln, invsigma = gexts.multihead_projection_layernorm_with_position_rrc_forward(
            input, weight, bias, lnw, lnb, norm_eps, position,
            seqlen, d_model, num_head, dim_head, dim_position, batch
        )
        ctx.save_for_backward(input, weight, hat_ln, invsigma, lnw, lnb)
        ctx.shape_kwargs = [num_head, dim_head, dim_position]
        # hat_ln = [batch, num_head, dim_position + dim_head, seqlen]
        return hat_ln

    @staticmethod
    def backward(ctx: Any, grad_ln: Tensor):
        input, weight, hat_ln, invsigma, lnw, lnb, = ctx.saved_tensors
        num_head, dim_head, dim_position = ctx.shape_kwargs
        batch, seqlen, d_model = input.size()
        [grad_input, grad_weight, grad_bias, grad_lnw, grad_lnb] = \
        gexts.multihead_projection_layernorm_with_position_rrc_backward(
            grad_ln, input, weight, hat_ln, invsigma, lnw, lnb,
            seqlen, d_model, num_head, dim_head, dim_position, batch
        )
        return grad_input, grad_weight, grad_bias, grad_lnw, grad_lnb, None, None, None, None, None


class Multihead_Galerkin_Attention_cccr_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, Q: Tensor, K: Tensor, V: Tensor):
        batch, num_head, d_posk, seqlen = Q.size()
        Attn = gexts.multihead_galerkin_attention_cccr_forward(Q, K, V, batch, num_head, d_posk, seqlen)
        ctx.save_for_backward(Q, K, V)
        # Attn = [batch, num_head, seqlen, d_posk]
        return Attn

    @staticmethod
    def backward(ctx: Any, grad_Attn):
        Q, K, V = ctx.saved_tensors
        batch, num_head, d_posk, seqlen = Q.size()
        grad_Q, grad_K, grad_V = gexts.multihead_galerkin_attention_cccr_backward(
            grad_Attn, Q, K, V, batch, num_head, d_posk, seqlen
        )
        return grad_Q, grad_K, grad_V


def multihead_projection_with_position_rrc_cuda(
    input: Tensor, weight: Tensor, bias: Tensor, position: Tensor,
    num_head, dim_head, dim_position
) -> Tensor:
    return Multihead_Porjection_with_Position_Function.apply(
        input, weight, bias, position, num_head, dim_head, dim_position
    )

def multihead_projection_layernorm_with_position_rrc_cuda(
    input: Tensor, weight: Tensor, bias: Tensor, ln_weight: Tensor, ln_bias: Tensor, position: Tensor,
    norm_eps, num_head, dim_head, dim_position
) -> Tensor:
    return Multihead_Porjection_Layernorm_with_Position_Function.apply(
        input, weight, bias, ln_weight, ln_bias, position, norm_eps, num_head, dim_head, dim_position
    )

def multihead_galerkin_attention_cccr_cuda(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    return Multihead_Galerkin_Attention_cccr_Function.apply(Q, K, V)

def batched_skinny_gemm(A: Tensor, B: Tensor, alpha: float = 1.0) -> Tensor:
    return gexts.batched_skinny_gemm(A, B, alpha)

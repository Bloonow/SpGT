import os
from typing import Any
import torch
import torch.utils.cpp_extension
from torch import Tensor
from SpGT.common.path import EXTENSION_PATH

dexts = torch.utils.cpp_extension.load(
    'datamove', os.path.join(EXTENSION_PATH, 'bind', 'datamove.cu'), verbose=True
)


class Batched_Transpose_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: Tensor, batch_list: list[int], M_list: list[int], N_list: list[int]):
        ctx.extras_shape = [batch_list, M_list, N_list]
        return dexts.batched_transpose(X, batch_list, M_list, N_list)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        batch_list, M_list, N_list = ctx.extras_shape
        grad_X = dexts.batched_transpose(grad_out.contiguous(), batch_list, N_list, M_list)
        return grad_X, None, None, None


class Transpose_Gather_Dual_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, src: Tensor, dim_list: list[int], H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
    ):
        ctx.extras_shape = [dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2]
        out1, out2 = dexts.transpose_gather_dual(
            src, dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
        )
        return out1, out2

    @staticmethod
    def backward(ctx: Any, grad_out1: Tensor, grad_out2: Tensor):
        dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2 = ctx.extras_shape
        grad_src = dexts.transpose_scatter_dual(
            grad_out1.contiguous(), grad_out2.contiguous(),
            dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
        )
        return grad_src, None, None, None, None, None, None, None, None, None


class Transpose_Scatter_Dual_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, src1: Tensor, src2: Tensor, dim_list: list[int],
        H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
    ):
        ctx.extras_shape = [dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2]
        out = dexts.transpose_scatter_dual(
            src1, src2, dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
        )
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2 = ctx.extras_shape
        grad_src1, grad_src2 = dexts.transpose_gather_dual(
            grad_out.contiguous(), dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
        )
        return grad_src1, grad_src2, None, None, None, None, None, None, None, None, None


def batched_transpose(
    X: Tensor, batch_list: list[int], M_list: list[int], N_list: list[int]
) -> Tensor:
    return Batched_Transpose_Function.apply(X, batch_list, M_list, N_list)


def transpose_gather_dual(
    src: Tensor, dim_list: list[int], H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
) -> tuple[Tensor, Tensor]:
    return Transpose_Gather_Dual_Function.apply(
        src, dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
    )


def transpose_scatter_dual(
    src1: Tensor, src2: Tensor, dim_list: list[int], H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
) -> Tensor:
    return Transpose_Scatter_Dual_Function.apply(
        src1, src2, dim_list, H_all, W_all, H_low, W_low, H_base_1, W_base_1, H_base_2, W_base_2
    )

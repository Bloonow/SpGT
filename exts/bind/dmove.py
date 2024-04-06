import os
from typing import Any
import torch
import torch.utils.cpp_extension
from torch import Tensor

from SpGT.common.path import EXTS_PATH
dmove = torch.utils.cpp_extension.load(
    'dmove', os.path.join(EXTS_PATH, 'bind', 'dmove.cu'), verbose=True
)


class Batched_Transpose_2D_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: Tensor, M_list: list[int], N_list: list[int], batch_list: list[int]):
        ctx.extras_shape = [batch_list, M_list, N_list]
        return dmove.batched_transpose_2D(X, M_list, N_list, batch_list)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        batch_list, M_list, N_list = ctx.extras_shape
        grad_X = dmove.batched_transpose_2D(grad_out.contiguous(), N_list, M_list, batch_list)
        return grad_X, None, None, None


class Gather_Transpose_dual_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, src: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2):
        ctx.extras_shape = [M_list, N0, N1, n0, n1, N1_offset1, N1_offset2]
        out1, out2 = dmove.gather_transpose_dual(src, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)
        return out1, out2

    @staticmethod
    def backward(ctx: Any, grad_out1: Tensor, grad_out2: Tensor):
        M_list, N0, N1, n0, n1, N1_offset1, N1_offset2 = ctx.extras_shape
        grad_src = dmove.scatter_transpose_dual(
            grad_out1.contiguous(), grad_out2.contiguous(), M_list, N0, N1, n0, n1, N1_offset1, N1_offset2
        )
        return grad_src, None, None, None, None, None, None, None


class Scatter_Transpose_dual_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, src1: Tensor, src2: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2):
        ctx.extras_shape = [M_list, N0, N1, n0, n1, N1_offset1, N1_offset2]
        out = dmove.scatter_transpose_dual(src1, src2, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        M_list, N0, N1, n0, n1, N1_offset1, N1_offset2 = ctx.extras_shape
        grad_src1, grad_src2 = dmove.gather_transpose_dual(grad_out.contiguous(), M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)
        return grad_src1, grad_src2, None, None, None, None, None, None, None


def batched_transpose_2D(
    X: Tensor, M_list: list[int], N_list: list[int], batch_list: list[int]
) -> Tensor:
    return Batched_Transpose_2D_Function.apply(X, M_list, N_list, batch_list)


def gather_transpose_dual(
    src: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2
) -> tuple[Tensor,Tensor]:
    return Gather_Transpose_dual_Function.apply(src, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)


def scatter_transpose_dual(
    src1: Tensor, src2: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2
) -> Tensor:
    return Scatter_Transpose_dual_Function.apply(src1, src2, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)

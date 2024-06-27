import os
from typing import Any
import torch
import torch.utils.cpp_extension
from torch import Tensor
from SpGT.common.path import EXTENSION_PATH

dmove = torch.utils.cpp_extension.load(
    'dmove', os.path.join(EXTENSION_PATH, 'bind', 'dmove.cu'), verbose=True
)


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


def gather_transpose_dual(
    src: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2
) -> tuple[Tensor,Tensor]:
    return Gather_Transpose_dual_Function.apply(src, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)


def scatter_transpose_dual(
    src1: Tensor, src2: Tensor, M_list: list[int], N0, N1, n0, n1, N1_offset1, N1_offset2
) -> Tensor:
    return Scatter_Transpose_dual_Function.apply(src1, src2, M_list, N0, N1, n0, n1, N1_offset1, N1_offset2)

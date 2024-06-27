import os
from typing import Any
import torch
import torch.utils.cpp_extension
from torch import Tensor
from SpGT.common.path import EXTENSION_PATH

exts = torch.utils.cpp_extension.load(
    'datamove', os.path.join(EXTENSION_PATH, 'bind', 'datamove.cu'), verbose=True
)


class Batched_Transpose_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: Tensor, batch_list: list[int], M_list: list[int], N_list: list[int]):
        ctx.extras_shape = [batch_list, M_list, N_list]
        return exts.batched_transpose(X, batch_list, M_list, N_list)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        batch_list, M_list, N_list = ctx.extras_shape
        grad_X = exts.batched_transpose(grad_out.contiguous(), batch_list, M_list, N_list)
        return grad_X, None, None, None


def batched_transpose(
    X: Tensor, batch_list: list[int], M_list: list[int], N_list: list[int]
) -> Tensor:
    return Batched_Transpose_Function.apply(X, batch_list, M_list, N_list)

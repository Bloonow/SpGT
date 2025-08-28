import math
import os
import torch
import torch.utils.benchmark as benchmark
from SpGT.common.path import EVALUATION_PATH
from SpGT.common.trivial import caller_name, get_daytime_string, timize

from SpGT.network.layer import SpectralConv2D
from SpGT.network.sp_layer import Sp_SpectralConv2D


def time_fno2d_wrt_resolution(
    resolution_list: list[int],
    batch=4, in_dim=128, out_dim=32,
):
    # 测试的所有情况 resolution_list = [32, 64, 128, 256, 512, 1024]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(resolution, ModuleClz):
        mode = math.ceil(math.sqrt(resolution))
        model: torch.nn.Module = ModuleClz(in_dim, out_dim, mode=mode, droprate=0.0, activation='silu')
        model = model.to(device)
        X = torch.rand([batch, resolution, resolution, in_dim], device=device, requires_grad=True)
        Y = model(X)
        G = torch.rand_like(Y)
        return model, X, Y, G

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 基准 正向过程
        model, X, Y, G = gen_data(resolution, ModuleClz=SpectralConv2D)
        kwargs = dict(X=X)
        orig_forward = timize(model, 'Orig_SpectralConv2D Forward', label, sublabel, **kwargs)
        # 基准 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_backward = timize(torch.autograd.grad, 'Orig_SpectralConv2D Backward', label, sublabel, **kwargs)

        # 优化 正向过程
        model, X, Y, G = gen_data(resolution, ModuleClz=Sp_SpectralConv2D)
        kwargs = dict(X=X)
        exts_forward = timize(model, 'Exts_SpectralConv2D Forward', label, sublabel, **kwargs)
        # 优化 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_backward = timize(torch.autograd.grad, 'Exts_SpectralConv2D Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(orig_forward)
        time_results.append(orig_backward)
        time_results.append(exts_forward)
        time_results.append(exts_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'in_dim          = {in_dim}' + os.linesep
    msg += f'out_dim         = {out_dim}' + os.linesep
    msg += f'batch           = {batch}' + os.linesep
    msg += f'resolution_list = {resolution_list}' + os.linesep
    msg += f'modes_list      = {[math.ceil(math.sqrt(r)) for r in resolution_list]}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(EVALUATION_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_fno2d_wrt_batch(
    batch_list: list[int],
    resolution=256, in_dim=128, out_dim=32,
):
    # 测试的所有情况 batch_list = [2, 4, 8, 16, 32, 64]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(batch, ModuleClz):
        mode = math.ceil(math.sqrt(resolution))
        model: torch.nn.Module = ModuleClz(in_dim, out_dim, mode=mode, droprate=0.0, activation='silu')
        model = model.to(device)
        X = torch.rand([batch, resolution, resolution, in_dim], device=device, requires_grad=True)
        Y = model(X)
        G = torch.rand_like(Y)
        return model, X, Y, G

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 基准 正向过程
        model, X, Y, G = gen_data(batch, ModuleClz=SpectralConv2D)
        kwargs = dict(X=X)
        orig_forward = timize(model, 'Orig_SpectralConv2D Forward', label, sublabel, **kwargs)
        # 基准 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_backward = timize(torch.autograd.grad, 'Orig_SpectralConv2D Backward', label, sublabel, **kwargs)

        # 优化 正向过程
        model, X, Y, G = gen_data(batch, ModuleClz=Sp_SpectralConv2D)
        kwargs = dict(X=X)
        exts_forward = timize(model, 'Exts_SpectralConv2D Forward', label, sublabel, **kwargs)
        # 优化 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_backward = timize(torch.autograd.grad, 'Exts_SpectralConv2D Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(orig_forward)
        time_results.append(orig_backward)
        time_results.append(exts_forward)
        time_results.append(exts_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'in_dim     = {in_dim}' + os.linesep
    msg += f'out_dim    = {out_dim}' + os.linesep
    msg += f'resolution = {resolution}' + os.linesep
    msg += f'mode      = {math.ceil(math.sqrt(resolution))}' + os.linesep
    msg += f'batch_list = {batch_list}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(EVALUATION_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)

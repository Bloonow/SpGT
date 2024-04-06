import os
from typing import Callable
import torch
import torch.utils.benchmark as benchmark
from SpGT.common.path import TIME_PATH
from SpGT.common.trivial import caller_name, get_daytime_string, timize
from SpGT.exts.bind.galerkin import batched_skinny_gemm
from SpGT.exts.bind.galerkin import projpos_rrc_cuda, projpos_lnorm_rrc_cuda
from SpGT.module.layer import SimpleAttention

projpos_cuda:       Callable = projpos_rrc_cuda
projpos_lnorm_cuda: Callable = projpos_lnorm_rrc_cuda
projpos_orig:       Callable = SimpleAttention.projpos_orig
projpos_lnorm_orig: Callable = SimpleAttention.projpos_lnorm_orig

################################


def time_skinny_gemm_wrt_resolution(
    resolution_list: list[int],
    batch=16, d_posk=32
):
    # 测试的所有情况 resolution_list = [32, 64, 128, 256, 512, 1024]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(resolution):
        seqlen = resolution * resolution
        A = torch.rand([batch, d_posk, seqlen], device=device)
        B = torch.rand([batch, seqlen, d_posk], device=device)
        return A, B

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    A, B = gen_data(max(resolution_list))
    batched_skinny_gemm(A, B)

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 基准
        A, B = gen_data(resolution)
        kwargs = dict(input=A, other=B)
        orig_gemm = timize(torch.matmul, 'Orig Skinny GEMM', label, sublabel, **kwargs)

        # 优化
        A, B = gen_data(resolution)
        kwargs = dict(A=A, B=B)
        exts_gemm = timize(batched_skinny_gemm, 'Exts Skinny GEMM', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(orig_gemm)
        time_results.append(exts_gemm)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'd_posk          = {d_posk}' + os.linesep
    msg += f'batch           = {batch}' + os.linesep
    msg += f'resolution_list = {resolution_list}' + os.linesep
    msg += f'seqlen_list     = {[r * r for r in resolution_list]}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_skinny_gemm_wrt_batch(
    batch_list: list[int],
    resolution=256, d_posk=32
):
    # 测试的所有情况 batch_list = [4, 8, 16, 32, 64, 128, 256, 512]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(batch):
        seqlen = resolution * resolution
        A = torch.rand([batch, d_posk, seqlen], device=device)
        B = torch.rand([batch, seqlen, d_posk], device=device)
        return A, B

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    A, B = gen_data(max(batch_list))
    batched_skinny_gemm(A, B)

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 基准
        A, B = gen_data(batch)
        kwargs = dict(input=A, other=B)
        orig_gemm = timize(torch.matmul, 'Orig Skinny GEMM', label, sublabel, **kwargs)

        # 优化
        A, B = gen_data(batch)
        kwargs = dict(A=A, B=B)
        exts_gemm = timize(batched_skinny_gemm, 'Exts Skinny GEMM', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(orig_gemm)
        time_results.append(exts_gemm)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'd_posk     = {d_posk}' + os.linesep
    msg += f'resolution = {resolution}' + os.linesep
    msg += f'seqlen     = {resolution * resolution}' + os.linesep
    msg += f'batch_list = {batch_list}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_projpos_lnorm_wrt_resolution(
    resolution_list: list[int],
    batch=16, n_head=4, d_k=32, d_pos=2, norm_eps=1.e-5
):
    # 测试的所有情况 resolution_list = [32, 64, 128, 256, 512, 1024]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(resolution):
        seqlen = resolution * resolution
        d_model = n_head * d_k
        input = torch.rand([batch, seqlen, d_model], device=device, requires_grad=True)
        weight = torch.rand([d_model, d_model], device=device, requires_grad=True)
        bias = torch.rand([d_model], device=device, requires_grad=True)
        pos = torch.rand([batch, seqlen, d_pos], device=device)
        lnw = torch.rand([n_head, d_k], device=device, requires_grad=True)
        lnb = torch.rand([n_head, d_k], device=device, requires_grad=True)
        return input, weight, bias, pos, lnw, lnb

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # projpos 基准 正向过程
        input, weight, bias, pos, _, _ = gen_data(resolution)
        kwargs = dict(input=input, weight=weight, bias=bias, pos=pos, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_orig(**kwargs)
        G = torch.rand_like(Y)
        projpos_orig_forward = timize(projpos_orig, 'projpos_orig Forward', label, sublabel, **kwargs)
        # projpos 基准 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias], grad_outputs=G, retain_graph=True)
        projpos_orig_backward = timize(torch.autograd.grad, 'projpos_orig Backward', label, sublabel, **kwargs)

        # projpos 优化 正向过程
        input, weight, bias, pos, _, _ = gen_data(resolution)
        kwargs = dict(input=input, weight=weight, bias=bias, pos=pos, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_cuda(**kwargs)
        G = torch.rand_like(Y)
        projpos_cuda_forward = timize(projpos_cuda, 'projpos_cuda Forward', label, sublabel, **kwargs)
        # projpos 优化 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias], grad_outputs=G, retain_graph=True)
        projpos_cuda_backward = timize(torch.autograd.grad, 'projpos_cuda Backward', label, sublabel, **kwargs)

        # projpos_lnorm 基准 正向过程
        input, weight, bias, pos, lnw, lnb = gen_data(resolution)
        kwargs = dict(input=input, weight=weight, bias=bias, lnw=lnw, lnb=lnb, pos=pos, norm_eps=norm_eps, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_lnorm_orig(**kwargs)
        G = torch.rand_like(Y)
        projpos_lnorm_orig_forward = timize(projpos_lnorm_orig, 'projpos_lnorm_orig Forward', label, sublabel, **kwargs)
        # projpos_lnorm 基准 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias, lnw, lnb], grad_outputs=G, retain_graph=True)
        projpos_lnorm_orig_backward = timize(torch.autograd.grad, 'projpos_lnorm_orig Backward', label, sublabel, **kwargs)

        # projpos_lnorm 优化 正向过程
        input, weight, bias, pos, lnw, lnb = gen_data(resolution)
        kwargs = dict(input=input, weight=weight, bias=bias, lnw=lnw, lnb=lnb, pos=pos, norm_eps=norm_eps, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_lnorm_cuda(**kwargs)
        G = torch.rand_like(Y)
        projpos_lnorm_cuda_forward = timize(projpos_lnorm_cuda, 'projpos_lnorm_cuda Forward', label, sublabel, **kwargs)
        # projpos_lnorm 优化 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias, lnw, lnb], grad_outputs=G, retain_graph=True)
        projpos_lnorm_cuda_backward = timize(torch.autograd.grad, 'projpos_lnorm_cuda Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(projpos_orig_forward)
        time_results.append(projpos_orig_backward)
        time_results.append(projpos_cuda_forward)
        time_results.append(projpos_cuda_backward)
        time_results.append(projpos_lnorm_orig_forward)
        time_results.append(projpos_lnorm_orig_backward)
        time_results.append(projpos_lnorm_cuda_forward)
        time_results.append(projpos_lnorm_cuda_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'norm_eps        = {norm_eps}' + os.linesep
    msg += f'n_head          = {n_head}' + os.linesep
    msg += f'd_k             = {d_k}' + os.linesep
    msg += f'd_pos           = {d_pos}' + os.linesep
    msg += f'batch           = {batch}' + os.linesep
    msg += f'resolution_list = {resolution_list}' + os.linesep
    msg += f'seqlen_list     = {[r * r for r in resolution_list]}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_projpos_lnorm_wrt_batch(
    batch_list: list[int],
    resolution=256, n_head=4, d_k=32, d_pos=2, norm_eps=1.e-5
):
    # 测试的所有情况 batch_list = [2, 4, 8, 16, 32, 64]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 生成数据辅助函数
    def gen_data(batch):
        seqlen = resolution * resolution
        d_model = n_head * d_k
        input = torch.rand([batch, seqlen, d_model], device=device, requires_grad=True)
        weight = torch.rand([d_model, d_model], device=device, requires_grad=True)
        bias = torch.rand([d_model], device=device, requires_grad=True)
        pos = torch.rand([batch, seqlen, d_pos], device=device)
        lnw = torch.rand([n_head, d_k], device=device, requires_grad=True)
        lnb = torch.rand([n_head, d_k], device=device, requires_grad=True)
        return input, weight, bias, pos, lnw, lnb

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # projpos 基准 正向过程
        input, weight, bias, pos, _, _ = gen_data(batch)
        kwargs = dict(input=input, weight=weight, bias=bias, pos=pos, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_orig(**kwargs)
        G = torch.rand_like(Y)
        projpos_orig_forward = timize(projpos_orig, 'projpos_orig Forward', label, sublabel, **kwargs)
        # projpos 基准 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias], grad_outputs=G, retain_graph=True)
        projpos_orig_backward = timize(torch.autograd.grad, 'projpos_orig Backward', label, sublabel, **kwargs)

        # projpos 优化 正向过程
        input, weight, bias, pos, _, _ = gen_data(batch)
        kwargs = dict(input=input, weight=weight, bias=bias, pos=pos, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_cuda(**kwargs)
        G = torch.rand_like(Y)
        projpos_cuda_forward = timize(projpos_cuda, 'projpos_cuda Forward', label, sublabel, **kwargs)
        # projpos 优化 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias], grad_outputs=G, retain_graph=True)
        projpos_cuda_backward = timize(torch.autograd.grad, 'projpos_cuda Backward', label, sublabel, **kwargs)

        # projpos_lnorm 基准 正向过程
        input, weight, bias, pos, lnw, lnb = gen_data(batch)
        kwargs = dict(input=input, weight=weight, bias=bias, lnw=lnw, lnb=lnb, pos=pos, norm_eps=norm_eps, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_lnorm_orig(**kwargs)
        G = torch.rand_like(Y)
        projpos_lnorm_orig_forward = timize(projpos_lnorm_orig, 'projpos_lnorm_orig Forward', label, sublabel, **kwargs)
        # projpos_lnorm 基准 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias, lnw, lnb], grad_outputs=G, retain_graph=True)
        projpos_lnorm_orig_backward = timize(torch.autograd.grad, 'projpos_lnorm_orig Backward', label, sublabel, **kwargs)

        # projpos_lnorm 优化 正向过程
        input, weight, bias, pos, lnw, lnb = gen_data(batch)
        kwargs = dict(input=input, weight=weight, bias=bias, lnw=lnw, lnb=lnb, pos=pos, norm_eps=norm_eps, n_head=n_head, d_k=d_k, d_pos=d_pos)
        Y = projpos_lnorm_cuda(**kwargs)
        G = torch.rand_like(Y)
        projpos_lnorm_cuda_forward = timize(projpos_lnorm_cuda, 'projpos_lnorm_cuda Forward', label, sublabel, **kwargs)
        # projpos_lnorm 优化 反向过程
        kwargs = dict(outputs=Y, inputs=[input, weight, bias, lnw, lnb], grad_outputs=G, retain_graph=True)
        projpos_lnorm_cuda_backward = timize(torch.autograd.grad, 'projpos_lnorm_cuda Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(projpos_orig_forward)
        time_results.append(projpos_orig_backward)
        time_results.append(projpos_cuda_forward)
        time_results.append(projpos_cuda_backward)
        time_results.append(projpos_lnorm_orig_forward)
        time_results.append(projpos_lnorm_orig_backward)
        time_results.append(projpos_lnorm_cuda_forward)
        time_results.append(projpos_lnorm_cuda_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'norm_eps   = {norm_eps}' + os.linesep
    msg += f'n_head     = {n_head}' + os.linesep
    msg += f'd_k        = {d_k}' + os.linesep
    msg += f'd_pos      = {d_pos}' + os.linesep
    msg += f'resolution = {resolution}' + os.linesep
    msg += f'seqlen     = {resolution * resolution}' + os.linesep
    msg += f'batch_list = {batch_list}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)

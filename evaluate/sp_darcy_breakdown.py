import copy
import math
import os
import torch
import torch.utils.benchmark as benchmark
from SpGT.common.path import TIME_PATH
from SpGT.common.trivial import UnitGaussianNormalizer, caller_name, get_daytime_string, get_down_up_size, timize
from SpGT.module.model_exts import SimpleEncoderLayer_Exts, SpectralRegressor_Exts
from SpGT.module.model import DownScaler2D, SimpleEncoderLayer, SpectralRegressor, UpScaler2D


################################
# Breakdown 分析每个 component 组件时间
# DownScaler + EncoderLayers + UpScaler + FNO2D
################################


def get_darcy_config(resolution, batch):
    cfg = dict(
        # 这些参数可能随时更改，以配置不同的测试规模
        fine_resolution=resolution, batch_size=batch,
        subsample_node=1, subsample_attn=2,
        n_encoder_layer=6, n_regressor_layer=2, d_fourier_mode=12,
        num_worker=8, epochs=100, lr=0.001, metric_gamma=0.5,
        # 以下是固定参数，基本不会更改
        d_node=1, d_target=1, d_pos=2, boundary_condition='dirichlet', noise=0.0,
        downscaler_size=None, downscaler_activation='relu', downscaler_droprate=0.05,
        upscaler_size=None, upscaler_activation='silu', upscaler_droprate=0.0,
        encoder_droprate=0.0, decoder_droprate=0.0, gt_droprate=0.05,
        d_hidden=128,  n_encoder_head=4, d_encoder_ffn_hidden=256,
        init_xavier_uniform_gain=0.01, init_diagonal_weight=0.01, init_symmetric=False, norm_eps=1e-05,
        regressor_type='rfft2D', use_spacial_fc=True, d_frequency=32,
        regressor_activation='silu', seed=1127802,
    )
    R = cfg['fine_resolution']
    sub = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r = int((R - 1) / sub + 1)
    r_attn = int((R - 1) / sub_attn + 1)
    down_size, up_size = get_down_up_size(r, r_attn)
    cfg['downscaler_size'] = down_size
    cfg['upscaler_size'] = up_size
    cfg['d_fourier_mode'] = math.ceil(math.sqrt(r))
    cfg['target_normalizer'] = UnitGaussianNormalizer()
    return cfg


def gen_downscaler(resolution, batch):
    """
    Input  : [N, r_in, r_in, in_dim]
    Output : [N, r_out, r_out, out_dim]
    """
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution, batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    downscaler = DownScaler2D(
        cfg['d_node'], cfg['d_hidden'], cfg['downscaler_size'], cfg['downscaler_droprate'], cfg['downscaler_activation']
    ).to(device)
    X = torch.rand([batch, r, r, cfg['d_node']], requires_grad=True, device=device)
    Y = downscaler(X)
    G = torch.rand_like(Y)
    return downscaler, X, Y, G


def gen_upscaler(resolution, batch):
    """
    Input  : [N, r_in, r_in, in_dim]
    Output : [N, r_out, r_out, out_dim]
    """
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution, batch)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    upscaler = UpScaler2D(
        cfg['d_hidden'], cfg['d_hidden'], cfg['upscaler_size'], cfg['upscaler_droprate'], cfg['upscaler_activation']
    ).to(device)
    X = torch.rand([batch, r_attn, r_attn, cfg['d_hidden']], requires_grad=True, device=device)
    Y = upscaler(X)
    G = torch.rand_like(Y)
    return upscaler, X, Y, G


def gen_encoder_layers(resolution, batch, ModuleClz):
    """
    Inputs:
        X   : [N, seqlen, d_model]
        pod : [N, seqlen, d_pos]
    Output  : [N, seqlen, d_model]
    """
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution, batch)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    encoder = ModuleClz(
        cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
        cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
    )
    layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])]).to(device)

    def encoderlayers(X, pos):
        for layer in layers:
            X = layer(X, pos)
        return X
    X = torch.rand([batch, r_attn * r_attn, cfg['d_hidden']], requires_grad=True, device=device)
    pos = torch.rand([batch, r_attn * r_attn, cfg['d_pos']], device=device)
    Y = encoderlayers(X, pos=pos)
    G = torch.rand_like(Y)
    return encoderlayers, X, pos, Y, G


def gen_regressor(resolution, batch, ModuleClz):
    """
    Input  : [N, r, r, in_dim]
    Output : [N, r, r, out_dim]
    """
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution, batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    regressor = ModuleClz(
        cfg['d_hidden'], cfg['d_target'], cfg['d_frequency'], cfg['d_frequency'], cfg['n_regressor_layer'], cfg['d_fourier_mode'],
        cfg['use_spacial_fc'], cfg['d_pos'], cfg['decoder_droprate'], cfg['regressor_activation']
    ).to(device)
    X = torch.rand([batch, r, r, cfg['d_hidden']], requires_grad=True, device=device)
    grid = torch.rand([batch, r, r, cfg['d_pos']], device=device)
    Y = regressor(X, grid=grid)
    G = torch.rand_like(Y)
    return regressor, X, grid, Y, G


################################


def time_darcy_breakdown_wrt_resolution(
    resolution_list: list[int], batch=4,
):
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    gen_encoder_layers(max(resolution_list), batch, SimpleEncoderLayer_Exts)

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # downscaler 正向过程
        downscaler, X, Y, G = gen_downscaler(resolution, batch)
        kwargs = dict(X=X)
        downscaler_forward = timize(downscaler, 'DownScaler Forward', label, sublabel, **kwargs)
        # downscaler 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        downscaler_backward = timize(torch.autograd.grad, 'DownScaler Backward', label, sublabel, **kwargs)

        # upscaler 正向过程
        upscaler, X, Y, G = gen_upscaler(resolution, batch)
        kwargs = dict(X=X)
        upscaler_forward = timize(upscaler, 'UpScaler Forward', label, sublabel, **kwargs)
        # upscaler 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        upscaler_backward = timize(torch.autograd.grad, 'UpScaler Backward', label, sublabel, **kwargs)

        # orig_encoderlayers 正向过程
        orig_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, SimpleEncoderLayer)
        kwargs = dict(X=X, pos=pos)
        orig_encoderlayers_forward = timize(orig_encoderlayers, 'EncoderLayer_Orig Forward', label, sublabel, **kwargs)
        # orig_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Orig Backward', label, sublabel, **kwargs)

        # exts_encoderlayers 正向过程
        exts_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, SimpleEncoderLayer_Exts)
        kwargs = dict(X=X, pos=pos)
        exts_encoderlayers_forward = timize(exts_encoderlayers, 'EncoderLayer_Exts Forward', label, sublabel, **kwargs)
        # exts_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Exts Backward', label, sublabel, **kwargs)

        # orig_regressor 正向过程
        orig_regressor, X, grid, Y, G = gen_regressor(resolution, batch, SpectralRegressor)
        kwargs = dict(X=X, grid=grid)
        orig_regressor_forward = timize(orig_regressor, 'SpectralRegressor_Orig Forward', label, sublabel, **kwargs)
        # orig_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Orig Backward', label, sublabel, **kwargs)

        # exts_regressor 正向过程
        exts_regressor, X, grid, Y, G = gen_regressor(resolution, batch, SpectralRegressor_Exts)
        kwargs = dict(X=X, grid=grid)
        exts_regressor_forward = timize(exts_regressor, 'SpectralRegressor_Exts Forward', label, sublabel, **kwargs)
        # exts_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Exts Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(downscaler_forward)
        time_results.append(downscaler_backward)
        time_results.append(upscaler_forward)
        time_results.append(upscaler_backward)
        time_results.append(orig_encoderlayers_forward)
        time_results.append(orig_encoderlayers_backward)
        time_results.append(exts_encoderlayers_forward)
        time_results.append(exts_encoderlayers_backward)
        time_results.append(orig_regressor_forward)
        time_results.append(orig_regressor_backward)
        time_results.append(exts_regressor_forward)
        time_results.append(exts_regressor_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'batch           = {batch}' + os.linesep
    msg += f'resolution_list = {resolution_list}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_darcy_breakdown_wrt_batch(
    batch_list: list[int], resolution=128,
):
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    gen_encoder_layers(resolution, max(batch_list), SimpleEncoderLayer_Exts)

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # downscaler 正向过程
        downscaler, X, Y, G = gen_downscaler(resolution, batch)
        kwargs = dict(X=X)
        downscaler_forward = timize(downscaler, 'DownScaler Forward', label, sublabel, **kwargs)
        # downscaler 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        downscaler_backward = timize(torch.autograd.grad, 'DownScaler Backward', label, sublabel, **kwargs)

        # upscaler 正向过程
        upscaler, X, Y, G = gen_upscaler(resolution, batch)
        kwargs = dict(X=X)
        upscaler_forward = timize(upscaler, 'UpScaler Forward', label, sublabel, **kwargs)
        # upscaler 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        upscaler_backward = timize(torch.autograd.grad, 'UpScaler Backward', label, sublabel, **kwargs)

        # orig_encoderlayers 正向过程
        orig_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, SimpleEncoderLayer)
        kwargs = dict(X=X, pos=pos)
        orig_encoderlayers_forward = timize(orig_encoderlayers, 'EncoderLayer_Orig Forward', label, sublabel, **kwargs)
        # orig_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Orig Backward', label, sublabel, **kwargs)

        # exts_encoderlayers 正向过程
        exts_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, SimpleEncoderLayer_Exts)
        kwargs = dict(X=X, pos=pos)
        exts_encoderlayers_forward = timize(exts_encoderlayers, 'EncoderLayer_Exts Forward', label, sublabel, **kwargs)
        # exts_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Exts Backward', label, sublabel, **kwargs)

        # orig_regressor 正向过程
        orig_regressor, X, grid, Y, G = gen_regressor(resolution, batch, SpectralRegressor)
        kwargs = dict(X=X, grid=grid)
        orig_regressor_forward = timize(orig_regressor, 'SpectralRegressor_Orig Forward', label, sublabel, **kwargs)
        # orig_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Orig Backward', label, sublabel, **kwargs)

        # exts_regressor 正向过程
        exts_regressor, X, grid, Y, G = gen_regressor(resolution, batch, SpectralRegressor_Exts)
        kwargs = dict(X=X, grid=grid)
        exts_regressor_forward = timize(exts_regressor, 'SpectralRegressor_Exts Forward', label, sublabel, **kwargs)
        # exts_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Exts Backward', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(downscaler_forward)
        time_results.append(downscaler_backward)
        time_results.append(upscaler_forward)
        time_results.append(upscaler_backward)
        time_results.append(orig_encoderlayers_forward)
        time_results.append(orig_encoderlayers_backward)
        time_results.append(exts_encoderlayers_forward)
        time_results.append(exts_encoderlayers_backward)
        time_results.append(orig_regressor_forward)
        time_results.append(orig_regressor_backward)
        time_results.append(exts_regressor_forward)
        time_results.append(exts_regressor_backward)

    compare = benchmark.Compare(results=time_results)
    hint = '' + caller_name() + '_' + get_daytime_string()
    msg = f'======== {torch.cuda.get_device_name(device)} ========' + os.linesep
    msg += f'resolution = {resolution}' + os.linesep
    msg += f'batch_list = {batch_list}' + os.linesep
    msg += f'======== {hint} ========' + os.linesep
    msg += f'{str(compare)}' + os.linesep
    print('', msg, sep=os.linesep)
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)

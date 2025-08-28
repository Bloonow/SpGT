import copy
import math
import os
import torch
import torch.utils.benchmark as benchmark
from SpGT.common.path import CONFIG_PATH, EVALUATION_PATH
from SpGT.common.trivial import caller_name, get_daytime_string, get_resolution_size, timize
from SpGT.config.config_accessor import read_config
from SpGT.dataset.darcy_dataset import GaussNormalizer
from SpGT.network.model import DownScaler2D, EncoderLayer, DecoderRegressor, UpScaler2D
from SpGT.network.sp_model import Sp_EncoderLayer, Sp_DecoderRegressor


################################
# Breakdown 分析每个 component 组件时间
# DownScaler + EncoderLayers + UpScaler + FNO2D
################################

def get_darcy_config(resolution, batch):
    filepath = os.path.join(CONFIG_PATH, 'darcy_config.yaml')
    cfg = read_config(filepath)
    cfg['fine_resolution'] = resolution
    cfg['batch_size'] = batch
    cfg['subsample_node'] = 1
    cfg['subsample_attn'] = 2

    # 调整超参数配置
    R = cfg['fine_resolution']
    sub_node = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r_node = int((R - 1) / sub_node + 1)
    r_attn = int((R - 1) / sub_attn + 1)
    # 设置降采样与升采样的值
    resolution_size = get_resolution_size(r_node, r_attn)
    cfg['resolution_size'] = resolution_size
    # 配置滤波所保留的频谱数目
    cfg['num_frequence_mode'] = math.ceil(math.sqrt(r_node))
    # 高斯正则化
    cfg['target_normalizer'] = GaussNormalizer().numpy_to_torch('cuda')
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
        cfg['resolution_size'], cfg['dim_node'], cfg['dim_hidden'], cfg['drop_downscaler'], cfg['acti_downscaler']
    ).to(device)
    X = torch.rand([batch, r, r, cfg['dim_node']], requires_grad=True, device=device)
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
        cfg['resolution_size'], cfg['dim_hidden'], cfg['dim_hidden'], cfg['drop_upscaler'], cfg['acti_upscaler']
    ).to(device)
    X = torch.rand([batch, r_attn, r_attn, cfg['dim_hidden']], requires_grad=True, device=device)
    Y = upscaler(X)
    G = torch.rand_like(Y)
    return upscaler, X, Y, G


def gen_encoder_layers(resolution, batch, ModuleClz):
    """
    Inputs:
        X   : [N, seqlen, d_model]
        pod : [N, seqlen, dim_position]
    Output  : [N, seqlen, d_model]
    """
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution, batch)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    encoder = ModuleClz(
        cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
        cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
        cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
    )
    layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['num_encoder_layer'])]).to(device)

    def encoderlayers(X, pos):
        for layer in layers:
            X = layer(X, pos)
        return X
    X = torch.rand([batch, r_attn * r_attn, cfg['dim_hidden']], requires_grad=True, device=device)
    pos = torch.rand([batch, r_attn * r_attn, cfg['dim_position']], device=device)
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
        cfg['num_decoder_layer'], cfg['dim_hidden'], cfg['dim_target'], cfg['dim_position'],
        cfg['dim_spatial_hidden'], cfg['num_frequence_mode'], cfg['drop_decoder_layer'], cfg['acti_decoder_layer']
    ).to(device)
    X = torch.rand([batch, r, r, cfg['dim_hidden']], requires_grad=True, device=device)
    grid = torch.rand([batch, r, r, cfg['dim_position']], device=device)
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
    gen_encoder_layers(max(resolution_list), batch, Sp_EncoderLayer)

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
        orig_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, EncoderLayer)
        kwargs = dict(X=X, pos=pos)
        orig_encoderlayers_forward = timize(orig_encoderlayers, 'EncoderLayer_Orig Forward', label, sublabel, **kwargs)
        # orig_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Orig Backward', label, sublabel, **kwargs)

        # exts_encoderlayers 正向过程
        exts_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, Sp_EncoderLayer)
        kwargs = dict(X=X, pos=pos)
        exts_encoderlayers_forward = timize(exts_encoderlayers, 'EncoderLayer_Exts Forward', label, sublabel, **kwargs)
        # exts_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Exts Backward', label, sublabel, **kwargs)

        # orig_regressor 正向过程
        orig_regressor, X, grid, Y, G = gen_regressor(resolution, batch, DecoderRegressor)
        kwargs = dict(X=X, grid=grid)
        orig_regressor_forward = timize(orig_regressor, 'SpectralRegressor_Orig Forward', label, sublabel, **kwargs)
        # orig_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Orig Backward', label, sublabel, **kwargs)

        # exts_regressor 正向过程
        exts_regressor, X, grid, Y, G = gen_regressor(resolution, batch, Sp_DecoderRegressor)
        kwargs = dict(X=X, grid=grid)
        exts_regressor_forward = timize(exts_regressor, 'Sp_DecoderRegressor Forward', label, sublabel, **kwargs)
        # exts_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_regressor_backward = timize(torch.autograd.grad, 'Sp_DecoderRegressor Backward', label, sublabel, **kwargs)

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
    path = os.path.join(EVALUATION_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_darcy_breakdown_wrt_batch(
    batch_list: list[int], resolution=128,
):
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    gen_encoder_layers(resolution, max(batch_list), Sp_EncoderLayer)

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
        orig_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, EncoderLayer)
        kwargs = dict(X=X, pos=pos)
        orig_encoderlayers_forward = timize(orig_encoderlayers, 'EncoderLayer_Orig Forward', label, sublabel, **kwargs)
        # orig_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Orig Backward', label, sublabel, **kwargs)

        # exts_encoderlayers 正向过程
        exts_encoderlayers, X, pos, Y, G = gen_encoder_layers(resolution, batch, Sp_EncoderLayer)
        kwargs = dict(X=X, pos=pos)
        exts_encoderlayers_forward = timize(exts_encoderlayers, 'EncoderLayer_Exts Forward', label, sublabel, **kwargs)
        # exts_encoderlayers 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_encoderlayers_backward = timize(torch.autograd.grad, 'EncoderLayer_Exts Backward', label, sublabel, **kwargs)

        # orig_regressor 正向过程
        orig_regressor, X, grid, Y, G = gen_regressor(resolution, batch, DecoderRegressor)
        kwargs = dict(X=X, grid=grid)
        orig_regressor_forward = timize(orig_regressor, 'SpectralRegressor_Orig Forward', label, sublabel, **kwargs)
        # orig_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        orig_regressor_backward = timize(torch.autograd.grad, 'SpectralRegressor_Orig Backward', label, sublabel, **kwargs)

        # exts_regressor 正向过程
        exts_regressor, X, grid, Y, G = gen_regressor(resolution, batch, Sp_DecoderRegressor)
        kwargs = dict(X=X, grid=grid)
        exts_regressor_forward = timize(exts_regressor, 'Sp_DecoderRegressor Forward', label, sublabel, **kwargs)
        # exts_regressor 反向过程
        kwargs = dict(outputs=Y, inputs=X, grad_outputs=G, retain_graph=True)
        exts_regressor_backward = timize(torch.autograd.grad, 'Sp_DecoderRegressor Backward', label, sublabel, **kwargs)

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
    path = os.path.join(EVALUATION_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)

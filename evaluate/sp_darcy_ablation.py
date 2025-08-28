import copy
import math
import os
from typing import Callable
import torch
import torch.utils.benchmark as benchmark
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from SpGT.common.path import CONFIG_PATH, EVALUATION_PATH
from SpGT.common.trivial import caller_name, get_daytime_string, get_resolution_size, timize
from SpGT.config.config_accessor import read_config
from SpGT.dataset.darcy_dataset import GaussNormalizer
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.extension.bind.galerkin_attention import multihead_galerkin_attention_cccr_cuda
from SpGT.extension.bind.galerkin_attention import multihead_projection_layernorm_with_position_rrc_cuda
from SpGT.extension.bind.galerkin_attention import multihead_projection_with_position_rrc_cuda
from SpGT.network.layer import GalerkinAttention
from SpGT.network.model import EncoderMixer, GalerkinTransformer2D, EncoderLayer
from SpGT.network.sp_model import Sp_GalerkinTransformer2D


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


def gen_train_data(resolution, batch):
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution=resolution, batch=batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    coeff = torch.rand([batch, r, r, 1], device=device)
    node = torch.rand([batch, r, r, 1], device=device)
    position = torch.rand([batch, r_attn, r_attn, 2], device=device)
    grid = torch.rand([batch, r, r, 2], device=device)
    target = torch.rand([batch, r, r, 1], device=device)
    target_grad = torch.rand([batch, r, r, 2], device=device)
    return coeff, node, position, grid, target, target_grad


def gen_valid_data(resolution, batch):
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution=resolution, batch=batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    node = torch.rand([batch, r, r, 1], device=device)
    position = torch.rand([batch, r_attn, r_attn, 2], device=device)
    grid = torch.rand([batch, r, r, 2], device=device)
    target = torch.rand([batch, r, r, 1], device=device)
    return node, position, grid, target


################################
# 消融实验
# V0 = baseline
# V1 = baseline + projpos
# V2 = baseline + projpos + projpos_lnorm
# V3 = baseline + projpos + projpos_lnorm + skinny_gemm
# V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
################################


class SimpleAttention_V1(GalerkinAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, position: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = multihead_projection_with_position_rrc_cuda(
            Q, self.weight_Q, self.bias_Q, position, self.num_head, self.dim_head, self.dim_position).transpose(-2, -1)
        K = self.multihead_projection_layernorm_with_position(
            K, self.weight_K, self.bias_K, self.ln_weight_K, self.ln_bias_K, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        V = self.multihead_projection_layernorm_with_position(
            V, self.weight_V, self.bias_V, self.ln_weight_V, self.ln_bias_V, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        # [N, num_head, seqlen, dim_position + dim_head]
        Attn = self.multihead_galerkin_attention(Q, K, V, droprate=0.0)
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.num_head * (self.dim_position + self.dim_head))
        output = self.fc_layer(Attn)
        return output


class SimpleAttention_V2(GalerkinAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, position: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = multihead_projection_with_position_rrc_cuda(
            Q, self.weight_Q, self.bias_Q, position, self.num_head, self.dim_head, self.dim_position).transpose(-2, -1)
        K = multihead_projection_layernorm_with_position_rrc_cuda(
            K, self.weight_K, self.bias_K, self.ln_weight_K, self.ln_bias_K, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        ).transpose(-2, -1)
        V = multihead_projection_layernorm_with_position_rrc_cuda(
            V, self.weight_V, self.bias_V, self.ln_weight_V, self.ln_bias_V, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        ).transpose(-2, -1)
        # [N, num_head, seqlen, dim_position + dim_head]
        Attn = self.multihead_galerkin_attention(Q, K, V, droprate=0.0)
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.num_head * (self.dim_position + self.dim_head))
        output = self.fc_layer(Attn)
        return output


class SimpleAttention_V3(GalerkinAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, position: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = multihead_projection_with_position_rrc_cuda(
            Q, self.weight_Q, self.bias_Q, position, self.num_head, self.dim_head, self.dim_position)
        K = multihead_projection_layernorm_with_position_rrc_cuda(
            K, self.weight_K, self.bias_K, self.ln_weight_K, self.ln_bias_K, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        V = multihead_projection_layernorm_with_position_rrc_cuda(
            V, self.weight_V, self.bias_V, self.ln_weight_V, self.ln_bias_V, position, self.norm_eps, self.num_head, self.dim_head, self.dim_position
        )
        Attn = multihead_galerkin_attention_cccr_cuda(Q, K, V)  # [N, num_head, seqlen, dim_position + dim_head]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.num_head * (self.dim_position + self.dim_head))
        output = self.fc_layer(Attn)
        return output


class SimpleEncoderLayer_V1(EncoderLayer):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.attention_layer = SimpleAttention_V1(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric
        )


class SimpleEncoderLayer_V2(EncoderLayer):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.attention_layer = SimpleAttention_V2(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric
        )


class SimpleEncoderLayer_V3(EncoderLayer):
    def __init__(
        self, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.attention_layer = SimpleAttention_V3(
            dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric
        )


class EncoderMixer_V1(EncoderMixer):
    def __init__(
        self, num_layer,
        dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            num_layer, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate,
            xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.layers = torch.nn.ModuleList([
            SimpleEncoderLayer_V1(
                dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
                ffn_dim_hidden, ffn_droprate, ffn_activation
            ) for _ in range(num_layer)
        ])


class EncoderMixer_V2(EncoderMixer):
    def __init__(
        self, num_layer,
        dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            num_layer, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate,
            xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.layers = torch.nn.ModuleList([
            SimpleEncoderLayer_V2(
                dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
                ffn_dim_hidden, ffn_droprate, ffn_activation
            ) for _ in range(num_layer)
        ])


class EncoderMixer_V3(EncoderMixer):
    def __init__(
        self, num_layer,
        dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
        ffn_dim_hidden, ffn_droprate, ffn_activation
    ) -> None:
        super().__init__(
            num_layer, dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate,
            xavier, diagonal, symmetric,
            ffn_dim_hidden, ffn_droprate, ffn_activation
        )
        self.layers = torch.nn.ModuleList([
            SimpleEncoderLayer_V3(
                dim_hidden, num_head, dim_position, norm_eps, attn_droprate, fc_droprate, xavier, diagonal, symmetric,
                ffn_dim_hidden, ffn_droprate, ffn_activation
            ) for _ in range(num_layer)
        ])


class ModuleClz_V1(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.encoder_mixer = EncoderMixer_V1(
            cfg['num_encoder_layer'], cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
            cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
            cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
        )


class ModuleClz_V2(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.encoder_mixer = EncoderMixer_V2(
            cfg['num_encoder_layer'], cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
            cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
            cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
        )


class ModuleClz_V3(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.encoder_mixer = EncoderMixer_V3(
            cfg['num_encoder_layer'], cfg['dim_hidden'], cfg['num_head'], cfg['dim_position'], cfg['attn_norm_eps'],
            cfg['drop_encoder_attn'], cfg['drop_encoder_attn_fc'], cfg['attn_xavier'], cfg['attn_diagonal'],
            cfg['attn_symmetric'], cfg['dim_encoder_ffn'], cfg['drop_encoder_ffn'], cfg['acti_encoder_ffn']
        )


ModuleClz_V0 = GalerkinTransformer2D
ModuleClz_V4 = Sp_GalerkinTransformer2D


def get_darcy_model(resolution, batch, ModuleClz):
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution=resolution, batch=batch)
    # 构造模型及其训练配置
    model: torch.nn.Module = ModuleClz(cfg)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], epochs=cfg['epochs'], steps_per_epoch=1024,
        pct_start=0.3, div_factor=1e4, final_div_factor=1e4
    )
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    train_loss_func = WeightedL2Loss2D(is_regularization=True, S=r, gamma=cfg['metric_gamma'])
    valid_loss_func = WeightedL2Loss2D(is_regularization=False, S=r)
    return model, optimizer, lr_scheduler, train_loss_func, valid_loss_func


def train_batch_darcy(
    node: Tensor, position: Tensor, grid: Tensor, coeff: Tensor, target: Tensor, target_grad: Tensor,
    model: torch.nn.Module, train_loss_func: Callable, optimizer: Optimizer,
    lr_scheduler: LRScheduler, grad_clip=0.999
):
    optimizer.zero_grad(set_to_none=True)
    pred = model(node, position=position, grid=grid)
    loss, reg, _ = train_loss_func(pred, target, pred_grad=None, target_grad=target_grad, coeff=coeff)
    loss = loss + reg
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    lr_scheduler.step()
    return loss


def validate_batch_darcy(
    node: Tensor, position: Tensor, grid: Tensor, target: Tensor,
    model: torch.nn.Module, valid_loss_func: Callable
):
    with torch.no_grad():
        pred = model(node, position=position, grid=grid)
        _, _, metric = valid_loss_func(pred, target, pred_grad=None, target_grad=None, coeff=None)
    return metric


################################


def time_darcy_ablation_wrt_resolution(
    resolution_list: list[int], batch=4
):
    # 测试的所有情况 resolution_list = [32, 64, 128, 256, 512, 1024]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    coeff, node, position, grid, target, target_grad = gen_train_data(max(resolution_list), batch)
    model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(max(resolution_list), batch, ModuleClz_V4)
    train_batch_darcy(node, position, grid, coeff, target, target_grad, model, train_loss_func, optimizer, lr_scheduler)

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 训练用数据
        coeff, node, position, grid, target, target_grad = gen_train_data(resolution, batch)
        kwargs = dict(node=node, position=position, grid=grid, coeff=coeff, target=target, target_grad=target_grad,)
        # 训练 V0 = baseline
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V0)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V0 = timize(train_batch_darcy, 'Train V0', label, sublabel, **kwargs)
        # 训练 V1 = baseline + projpos
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V1)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V1 = timize(train_batch_darcy, 'Train V1', label, sublabel, **kwargs)
        # 训练 V2 = baseline + projpos + projpos_lnorm
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V2)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V2 = timize(train_batch_darcy, 'Train V2', label, sublabel, **kwargs)
        # 训练 V3 = baseline + projpos + projpos_lnorm + skinny_gemm
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V3)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V3 = timize(train_batch_darcy, 'Train V3', label, sublabel, **kwargs)
        # 训练 V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V4)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V4 = timize(train_batch_darcy, 'Train V4', label, sublabel, **kwargs)

        # 推理用数据
        node, position, grid, target = gen_valid_data(resolution, batch)
        kwargs = dict(node=node, position=position, grid=grid, target=target,)
        # 推理 V0 = baseline
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V0)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V0 = timize(validate_batch_darcy, 'Valid V0', label, sublabel, **kwargs)
        # 推理 V1 = baseline + projpos
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V1)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V1 = timize(validate_batch_darcy, 'Valid V1', label, sublabel, **kwargs)
        # 推理 V2 = baseline + projpos + projpos_lnorm
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V2)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V2 = timize(validate_batch_darcy, 'Valid V2', label, sublabel, **kwargs)
        # 推理 V3 = baseline + projpos + projpos_lnorm + skinny_gemm
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V3)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V3 = timize(validate_batch_darcy, 'Valid V3', label, sublabel, **kwargs)
        # 推理 V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V4)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V4 = timize(validate_batch_darcy, 'Valid V4', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(train_V0)
        time_results.append(train_V1)
        time_results.append(train_V2)
        time_results.append(train_V3)
        time_results.append(train_V4)
        time_results.append(valid_V0)
        time_results.append(valid_V1)
        time_results.append(valid_V2)
        time_results.append(valid_V3)
        time_results.append(valid_V4)

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


def time_darcy_ablation_wrt_batch(
    batch_list: list[int], resolution=128
):
    # 测试的所有情况 batch_list = [2, 4, 8, 16, 32, 64]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    coeff, node, position, grid, target, target_grad = gen_train_data(resolution, max(batch_list))
    model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, max(batch_list), ModuleClz_V4)
    train_batch_darcy(node, position, grid, coeff, target, target_grad, model, train_loss_func, optimizer, lr_scheduler)

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 训练用数据
        coeff, node, position, grid, target, target_grad = gen_train_data(resolution, batch)
        kwargs = dict(node=node, position=position, grid=grid, coeff=coeff, target=target, target_grad=target_grad,)
        # 训练 V0 = baseline
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V0)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V0 = timize(train_batch_darcy, 'Train V0', label, sublabel, **kwargs)
        # 训练 V1 = baseline + projpos
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V1)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V1 = timize(train_batch_darcy, 'Train V1', label, sublabel, **kwargs)
        # 训练 V2 = baseline + projpos + projpos_lnorm
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V2)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V2 = timize(train_batch_darcy, 'Train V2', label, sublabel, **kwargs)
        # 训练 V3 = baseline + projpos + projpos_lnorm + skinny_gemm
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V3)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V3 = timize(train_batch_darcy, 'Train V3', label, sublabel, **kwargs)
        # 训练 V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
        model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, batch, ModuleClz_V4)
        kwargs.update(model=model, train_loss_func=train_loss_func, optimizer=optimizer, lr_scheduler=lr_scheduler)
        train_V4 = timize(train_batch_darcy, 'Train V4', label, sublabel, **kwargs)

        # 推理用数据
        node, position, grid, target = gen_valid_data(resolution, batch)
        kwargs = dict(node=node, position=position, grid=grid, target=target,)
        # 推理 V0 = baseline
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V0)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V0 = timize(validate_batch_darcy, 'Valid V0', label, sublabel, **kwargs)
        # 推理 V1 = baseline + projpos
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V1)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V1 = timize(validate_batch_darcy, 'Valid V1', label, sublabel, **kwargs)
        # 推理 V2 = baseline + projpos + projpos_lnorm
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V2)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V2 = timize(validate_batch_darcy, 'Valid V2', label, sublabel, **kwargs)
        # 推理 V3 = baseline + projpos + projpos_lnorm + skinny_gemm
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V3)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V3 = timize(validate_batch_darcy, 'Valid V3', label, sublabel, **kwargs)
        # 推理 V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
        model, _, _, _, valid_loss_func = get_darcy_model(resolution, batch, ModuleClz_V4)
        kwargs.update(model=model, valid_loss_func=valid_loss_func)
        valid_V4 = timize(validate_batch_darcy, 'Valid V4', label, sublabel, **kwargs)

        # 收集测试结果
        time_results.append(train_V0)
        time_results.append(train_V1)
        time_results.append(train_V2)
        time_results.append(train_V3)
        time_results.append(train_V4)
        time_results.append(valid_V0)
        time_results.append(valid_V1)
        time_results.append(valid_V2)
        time_results.append(valid_V3)
        time_results.append(valid_V4)

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

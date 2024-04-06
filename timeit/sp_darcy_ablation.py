import copy
import math
import os
from typing import Callable
import torch
import torch.utils.benchmark as benchmark
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from SpGT.common.path import TIME_PATH
from SpGT.common.trivial import UnitGaussianNormalizer, caller_name, get_daytime_string, get_down_up_size, timize
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.exts.bind.galerkin import galattn_cccr_cuda, projpos_lnorm_rrc_cuda, projpos_rrc_cuda
from SpGT.module.layer import SimpleAttention
from SpGT.module.model import GalerkinTransformer2D, SimpleEncoderLayer
from SpGT.module.model_exts import GalerkinTransformer2D_Exts


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


def gen_train_data(resolution, batch):
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution=resolution, batch=batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    coeff = torch.rand([batch, r, r, 1], device=device)
    node = torch.rand([batch, r, r, 1], device=device)
    pos = torch.rand([batch, r_attn, r_attn, 2], device=device)
    grid = torch.rand([batch, r, r, 2], device=device)
    target = torch.rand([batch, r, r, 1], device=device)
    target_grad = torch.rand([batch, r, r, 2], device=device)
    return coeff, node, pos, grid, target, target_grad


def gen_valid_data(resolution, batch):
    device = torch.cuda.current_device()
    cfg = get_darcy_config(resolution=resolution, batch=batch)
    r = int((cfg['fine_resolution'] - 1) / cfg['subsample_node'] + 1)
    r_attn = int((cfg['fine_resolution'] - 1) / cfg['subsample_attn'] + 1)
    node = torch.rand([batch, r, r, 1], device=device)
    pos = torch.rand([batch, r_attn, r_attn, 2], device=device)
    grid = torch.rand([batch, r, r, 2], device=device)
    target = torch.rand([batch, r, r, 1], device=device)
    return node, pos, grid, target


################################
# 消融实验
# V0 = baseline
# V1 = baseline + projpos
# V2 = baseline + projpos + projpos_lnorm
# V3 = baseline + projpos + projpos_lnorm + skinny_gemm
# V4 = baseline + projpos + projpos_lnorm + skinny_gemm + fno2d
################################


class SimpleAttention_V1(SimpleAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = projpos_rrc_cuda(Q, self.weight_Q, self.bias_Q, pos, self.n_head, self.d_k, self.d_pos).transpose(-2, -1)
        K = self.projpos_lnorm_orig(
            K, self.weight_K, self.bias_K, self.lnw_K, self.lnb_K, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        V = self.projpos_lnorm_orig(
            V, self.weight_V, self.bias_V, self.lnw_V, self.lnb_V, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        Attn = self.galattn_orig(Q, K, V)  # [N, n_head, seqlen, d_pos + d_k]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.n_head * (self.d_pos + self.d_k))
        output = self.fc_layer(Attn)
        return output


class SimpleAttention_V2(SimpleAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = projpos_rrc_cuda(Q, self.weight_Q, self.bias_Q, pos, self.n_head, self.d_k, self.d_pos).transpose(-2, -1)
        K = projpos_lnorm_rrc_cuda(
            K, self.weight_K, self.bias_K, self.lnw_K, self.lnb_K, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        ).transpose(-2, -1)
        V = projpos_lnorm_rrc_cuda(
            V, self.weight_V, self.bias_V, self.lnw_V, self.lnb_V, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        ).transpose(-2, -1)
        Attn = self.galattn_orig(Q, K, V)  # [N, n_head, seqlen, d_pos + d_k]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.n_head * (self.d_pos + self.d_k))
        output = self.fc_layer(Attn)
        return output


class SimpleAttention_V3(SimpleAttention):
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos: Tensor = None):
        # Q,K,V : [N, seqlen, d_model]
        N, seqlen, _ = Q.size()
        Q = projpos_rrc_cuda(Q, self.weight_Q, self.bias_Q, pos, self.n_head, self.d_k, self.d_pos)
        K = projpos_lnorm_rrc_cuda(
            K, self.weight_K, self.bias_K, self.lnw_K, self.lnb_K, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        V = projpos_lnorm_rrc_cuda(
            V, self.weight_V, self.bias_V, self.lnw_V, self.lnb_V, pos, self.norm_eps, self.n_head, self.d_k, self.d_pos
        )
        Attn = galattn_cccr_cuda(Q, K, V)  # [N, n_head, seqlen, d_pos + d_k]
        Attn = Attn.transpose(1, 2).contiguous().view(N, seqlen, self.n_head * (self.d_pos + self.d_k))
        output = self.fc_layer(Attn)
        return output


class SimpleEncoderLayer_V1(SimpleEncoderLayer):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
        init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
    ) -> None:
        super().__init__(
            d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
            init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
        )
        self.attention_layer = SimpleAttention_V1(
            d_model=d_model, n_head=n_head, d_pos=d_pos, norm_eps=norm_eps,
            init_xavier_uniform_gain=init_xavier_uniform_gain, init_diagonal_weight=init_diagonal_weight,
            init_symmetric=init_symmetric, droprate=droprate
        )


class SimpleEncoderLayer_V2(SimpleEncoderLayer):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
        init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
    ) -> None:
        super().__init__(
            d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
            init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
        )
        self.attention_layer = SimpleAttention_V2(
            d_model=d_model, n_head=n_head, d_pos=d_pos, norm_eps=norm_eps,
            init_xavier_uniform_gain=init_xavier_uniform_gain, init_diagonal_weight=init_diagonal_weight,
            init_symmetric=init_symmetric, droprate=droprate
        )


class SimpleEncoderLayer_V3(SimpleEncoderLayer):
    def __init__(
        self, d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
        init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
    ) -> None:
        super().__init__(
            d_model, n_head, d_pos, norm_eps, d_encoder_ffn_hidden,
            init_xavier_uniform_gain, init_diagonal_weight, init_symmetric, droprate
        )
        self.attention_layer = SimpleAttention_V3(
            d_model=d_model, n_head=n_head, d_pos=d_pos, norm_eps=norm_eps,
            init_xavier_uniform_gain=init_xavier_uniform_gain, init_diagonal_weight=init_diagonal_weight,
            init_symmetric=init_symmetric, droprate=droprate
        )


class ModuleClz_V1(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        encoder = SimpleEncoderLayer_V1(
            cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
            cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
        )
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])])


class ModuleClz_V2(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        encoder = SimpleEncoderLayer_V2(
            cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
            cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
        )
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])])


class ModuleClz_V3(GalerkinTransformer2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        encoder = SimpleEncoderLayer_V3(
            cfg['d_hidden'], cfg['n_encoder_head'], cfg['d_pos'], cfg['norm_eps'], cfg['d_encoder_ffn_hidden'],
            cfg['init_xavier_uniform_gain'], cfg['init_diagonal_weight'], cfg['init_symmetric'], cfg['encoder_droprate']
        )
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(cfg['n_encoder_layer'])])


ModuleClz_V0 = GalerkinTransformer2D
ModuleClz_V4 = GalerkinTransformer2D_Exts


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
    node: Tensor, pos: Tensor, grid: Tensor, coeff: Tensor, target: Tensor, target_grad: Tensor,
    model: torch.nn.Module, train_loss_func: Callable, optimizer: Optimizer,
    lr_scheduler: LRScheduler, grad_clip=0.999
):
    optimizer.zero_grad(set_to_none=True)
    pred = model(node, pos=pos, grid=grid)
    loss, reg, _ = train_loss_func(pred, target, pred_grad=None, target_grad=target_grad, coeff=coeff)
    loss = loss + reg
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    lr_scheduler.step()
    return loss


def validate_batch_darcy(
    node: Tensor, pos: Tensor, grid: Tensor, target: Tensor,
    model: torch.nn.Module, valid_loss_func: Callable
):
    with torch.no_grad():
        pred = model(node, pos=pos, grid=grid)
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
    coeff, node, pos, grid, target, target_grad = gen_train_data(max(resolution_list), batch)
    model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(max(resolution_list), batch, ModuleClz_V4)
    train_batch_darcy(node, pos, grid, coeff, target, target_grad, model, train_loss_func, optimizer, lr_scheduler)

    # 存储测试结果
    time_results = []
    for resolution in resolution_list:
        label = f'[batch = {batch}]'
        sublabel = f'[resolution = {resolution}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 训练用数据
        coeff, node, pos, grid, target, target_grad = gen_train_data(resolution, batch)
        kwargs = dict(node=node, pos=pos, grid=grid, coeff=coeff, target=target, target_grad=target_grad,)
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
        node, pos, grid, target = gen_valid_data(resolution, batch)
        kwargs = dict(node=node, pos=pos, grid=grid, target=target,)
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
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)


def time_darcy_ablation_wrt_batch(
    batch_list: list[int], resolution=128
):
    # 测试的所有情况 batch_list = [2, 4, 8, 16, 32, 64]
    device = torch.cuda.current_device()
    print(f'======== Timing {caller_name()} at {torch.cuda.get_device_name(device)} ========')

    # 令 GlobalBuffer 提前分配最大空间，以避免重分配
    coeff, node, pos, grid, target, target_grad = gen_train_data(resolution, max(batch_list))
    model, optimizer, lr_scheduler, train_loss_func, _ = get_darcy_model(resolution, max(batch_list), ModuleClz_V4)
    train_batch_darcy(node, pos, grid, coeff, target, target_grad, model, train_loss_func, optimizer, lr_scheduler)

    # 存储测试结果
    time_results = []
    for batch in batch_list:
        label = f'[resolution = {resolution}]'
        sublabel = f'[batch = {batch}]'
        print(f'======== Timing {label}, {sublabel} ========')

        # 训练用数据
        coeff, node, pos, grid, target, target_grad = gen_train_data(resolution, batch)
        kwargs = dict(node=node, pos=pos, grid=grid, coeff=coeff, target=target, target_grad=target_grad,)
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
        node, pos, grid, target = gen_valid_data(resolution, batch)
        kwargs = dict(node=node, pos=pos, grid=grid, target=target,)
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
    path = os.path.join(TIME_PATH, f'{hint}.txt')
    with open(path, mode='w+', encoding='utf=8') as f:
        f.write(msg)

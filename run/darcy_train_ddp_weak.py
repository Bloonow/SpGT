import os
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler
from SpGT.common.path import CONFIG_PATH, DATA_PATH, MODEL_PATH
from SpGT.common.trivial import distributed_initialize, get_daytime_string, get_num_params, is_main_process, set_seed
from SpGT.config.config_accessor import get_darcy_config
from SpGT.dataset.darcy_dataset import DarcyDataset
from SpGT.engine.darcy_engine import train_epoch_darcy, validate_epoch_darcy
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.engine.train import run_train_ddp
from SpGT.module.model import GalerkinTransformer2D
from SpGT.module.model_exts import GalerkinTransformer2D_Exts


def darcy_train_ddp(cfg):
    # 验证要训练的模型是否存在
    model_name: str = cfg['model_name']
    if model_name.endswith('exts'):
        ModuleClz = GalerkinTransformer2D_Exts
    elif model_name.endswith('orig'):
        ModuleClz = GalerkinTransformer2D
    else:
        raise NameError(f'The Module "{model_name}" Not Exist')

    # 配置一些超参数
    device = torch.cuda.current_device()
    set_seed(cfg['seed'])
    R = cfg['fine_resolution']
    sub = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r = int((R - 1) / sub + 1)
    r_attn = int((R - 1) / sub_attn + 1)

    # 构建数据集
    train_path = os.path.join(DATA_PATH, cfg['train_dataset'])
    valid_path = os.path.join(DATA_PATH, cfg['valid_dataset'])
    train_ds = DarcyDataset(
        train_path, num_data=cfg['num_data'], fine_resolution=R, subsample_node=sub, subsample_attn=sub_attn,
        is_training=True, noise=cfg['noise'],
    )
    sample_normalizer = train_ds.sample_normalizer
    target_normalizer = train_ds.target_normalizer
    valid_ds = DarcyDataset(
        valid_path, num_data=128, fine_resolution=R, subsample_node=sub, subsample_attn=sub_attn,
        is_training=False, noise=cfg['noise'], sample_normalizer=sample_normalizer
    )
    train_sampler, valid_sampler = None, None
    train_loader = DataLoader(train_ds, cfg['batch_size'], shuffle=True, num_workers=cfg['num_worker'], pin_memory=True)
    valid_loader = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, num_workers=cfg['num_worker'], pin_memory=True)
    data_sample = next(iter(train_loader))
    print('=' * 20, 'Data loader batch', '=' * 20)
    for key in data_sample.keys():
        print(key, "\t", data_sample[key].shape)
    print('=' * (40 + len('Data loader batch') + 2))

    # 构建 DDP 模型与训练配置
    torch.cuda.empty_cache()
    cfg['target_normalizer'] = target_normalizer.to(device)
    model = ModuleClz(cfg)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=[device])
    print(f"\nThe Numbers of Model's Parameters: {get_num_params(model)}\n")
    epochs = cfg['epochs']
    lr = cfg['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, div_factor=1e4, final_div_factor=1e4
    )
    train_loss_func = WeightedL2Loss2D(is_regularization=True, S=r, gamma=cfg['metric_gamma'])
    valid_loss_func = WeightedL2Loss2D(is_regularization=False, S=r)

    # 训练迭代
    time_start = time.time()
    checkpoint = run_train_ddp(
        model, train_loader, valid_loader, train_sampler, valid_sampler,
        train_epoch_darcy, validate_epoch_darcy, train_loss_func, valid_loss_func,
        optimizer, lr_scheduler, epochs=epochs, start_epoch=0, patience=20, device=device
    )
    time_end = time.time()
    print(f'========== train_time: {time_end - time_start:.6f} ==========')

    # 从 DDP 模型获得单进程模型，并由主进程保存结果
    model: torch.nn.Module = model.module
    # 主进程保存结果
    if is_main_process():
        save_path = os.path.join(MODEL_PATH, f'{model_name}_R{r}_R{r_attn}_{get_daytime_string()}.pt')
        checkpoint['Sample_Normalizer'] = sample_normalizer
        checkpoint['Target_Normalizer'] = target_normalizer
        checkpoint['Train_Config'] = cfg
        torch.save(checkpoint, save_path)
        print(f'========== Saving Results at {save_path} ==========')


if __name__ == '__main__':
    distributed_initialize()
    print('======== weak scaling ========')
    cfg = get_darcy_config()
    darcy_train_ddp(cfg)

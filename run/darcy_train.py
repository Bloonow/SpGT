import os
import time
import torch
from torch.utils.data.dataloader import DataLoader
from SpGT.common.path import DATA_PATH, MODEL_PATH
from SpGT.common.trivial import get_daytime_string, get_num_params, set_seed
from SpGT.config.config_accessor import get_darcy_config
from SpGT.dataset.darcy_dataset import DarcyDataset
from SpGT.engine.darcy_engine import train_epoch_darcy, validate_epoch_darcy
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.engine.train import run_train
from SpGT.network.model import GalerkinTransformer2D
from SpGT.network.sp_model import Sp_GalerkinTransformer2D


def darcy_train(cfg):
    if cfg['name_module'] == 'GT':
        ModuleClz = GalerkinTransformer2D
    elif cfg['name_module'] == 'SpGT':
        ModuleClz = Sp_GalerkinTransformer2D
    else:
        raise NotImplementedError(f"Module {cfg['name_module']} has no implementation")

    # 一些超参数
    device = torch.cuda.current_device()
    set_seed(cfg['seed'])
    R = cfg['fine_resolution']
    sub_node = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r = int((R - 1) / sub_node + 1)

    # 构建数据集
    train_path = os.path.join(DATA_PATH, cfg['train_dataset'])
    valid_path = os.path.join(DATA_PATH, cfg['valid_dataset'])
    train_ds = DarcyDataset(
        train_path, sub_node, sub_attn, cfg['num_data'], cfg['fine_resolution'], is_training=True,
        noise=cfg['noise'], random_seed=cfg['seed']
    )
    node_normalizer, target_normalizer = train_ds.node_normalizer, train_ds.target_normalizer
    valid_ds = DarcyDataset(
        valid_path, sub_node, sub_attn, 128, cfg['fine_resolution'], is_training=False,
        noise=cfg['noise'], random_seed=cfg['seed'], node_normalizer=node_normalizer
    )
    train_loader = DataLoader(
        train_ds, cfg['batch_size'], shuffle=True, num_workers=cfg['num_load_worker'], pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, 2 * cfg['batch_size'], shuffle=False, num_workers=cfg['num_load_worker'], pin_memory=True
    )
    eg = next(iter(train_loader))
    print('=' * 20, 'Data loader batch', '=' * 20)
    for key in eg.keys():
        # print(key, "\t", eg[key].shape)
        print(f'{key:<20}{eg[key].shape}')
    print('=' * (40 + len('Data loader batch') + 2))

    # 构建模型与训练配置
    torch.cuda.empty_cache()
    cfg['target_normalizer'] = target_normalizer.numpy_to_torch(device)
    model = ModuleClz(cfg)
    model = model.to(device)
    print(f"\nThe Numbers of {cfg['name_module']} Model's Parameters: {get_num_params(model)}\n")
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
    checkpoint = run_train(
        model, train_loader, valid_loader, train_epoch_darcy, validate_epoch_darcy, train_loss_func, valid_loss_func,
        optimizer, lr_scheduler, epochs=epochs, start_epoch=0, patience=20, device=device
    )
    time_end = time.time()
    print(f'========== train_time: {time_end - time_start:.6f} ==========')

    # 保存模型
    name, r, d, m = cfg['name_module'], r, cfg['dim_hidden'], cfg['num_frequence_mode']
    save_path = os.path.join(MODEL_PATH, f'{name}_r{r}d{d}m{m}_{get_daytime_string()}.pt')
    checkpoint['Train_Config'] = cfg
    torch.save(checkpoint, save_path)
    print(f'========== Saving Results at {save_path} ==========')


if __name__ == '__main__':
    cfg = get_darcy_config()
    darcy_train(cfg=cfg)

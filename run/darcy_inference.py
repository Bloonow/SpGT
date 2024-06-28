import os
import time
import torch
from torch.utils.data.dataloader import DataLoader
from SpGT.common.path import DATA_PATH, MODEL_PATH
from SpGT.common.trivial import get_num_params, set_seed
from SpGT.config.config_accessor import get_darcy_config
from SpGT.dataset.darcy_dataset import DarcyDataset, get_guass_normalizer
from SpGT.engine.darcy_engine import validate_epoch_darcy
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.network.model import GalerkinTransformer2D
from SpGT.network.sp_model import Sp_GalerkinTransformer2D


def darcy_inference(cfg, checkpoint):
    if cfg['name_module'] == 'GT':
        ModuleClz = GalerkinTransformer2D
    elif cfg['name_module'] == 'SpGT':
        ModuleClz = Sp_GalerkinTransformer2D
    else:
        raise NotImplementedError(f"Module {cfg['name_module']} has no implementation")

    # 配置一些超参数，推理时所采用的分辨率可能与训练时不同
    device = torch.cuda.current_device()
    set_seed(cfg['seed'])
    R = cfg['fine_resolution']
    sub_node = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r = int((R - 1) / sub_node + 1)
    r_attn = int((R - 1) / sub_attn + 1)

    # 构建数据集
    train_path = os.path.join(DATA_PATH, cfg['train_dataset'])
    valid_path = os.path.join(DATA_PATH, cfg['valid_dataset'])
    node_normalizer, target_normalizer = get_guass_normalizer(train_path, sub_node)
    valid_ds = DarcyDataset(
        valid_path, sub_node, sub_attn, cfg['num_data'], cfg['fine_resolution'], is_training=False,
        noise=cfg['noise'], random_seed=cfg['seed'], node_normalizer=node_normalizer
    )
    valid_loader = DataLoader(
        valid_ds, 2 * cfg['batch_size'], shuffle=False, num_workers=cfg['num_load_worker'], pin_memory=True
    )
    eg = next(iter(valid_loader))
    print('=' * 20, 'Data loader batch', '=' * 20)
    for key in eg.keys():
        # print(key, "\t", eg[key].shape)
        print(f'{key:<20}{eg[key].shape}')
    print('=' * (40 + len('Data loader batch') + 2))

    # 构建模型
    torch.cuda.empty_cache()
    cfg['target_normalizer'] = target_normalizer.numpy_to_torch(device)
    model = ModuleClz(cfg)
    model.load_state_dict(checkpoint['Module'])
    model = model.to(device)
    print(f"\nThe Numbers of {cfg['name_module']} Model's Parameters: {get_num_params(model)}\n")
    valid_loss_func = WeightedL2Loss2D(is_regularization=False, S=r)

    # 进行验证
    time_start = time.time()
    metric = validate_epoch_darcy(model, valid_loader, valid_loss_func, device=device).item()
    time_end = time.time()
    print(f'========== valid_time: {time_end - time_start:.6f} ==========')
    print(f'========== Metric : {metric} ==========')


if __name__ == '__main__':
    cfg = get_darcy_config()
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'SpGT_r141d128m12_0628_2228.pt'), map_location='cpu')
    darcy_inference(cfg=cfg, checkpoint=checkpoint)

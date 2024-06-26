import os
import time
import torch
from torch.utils.data.dataloader import DataLoader
from SpGT.common.path import DATA_PATH, MODEL_PATH
from SpGT.common.trivial import get_num_params, set_seed
from SpGT.config.config_accessor import get_darcy_config
from SpGT.dataset.darcy_dataset import DarcyDataset
from SpGT.engine.darcy_engine import validate_epoch_darcy
from SpGT.engine.metric import WeightedL2Loss2D
from SpGT.module.model import GalerkinTransformer2D
from SpGT.module.model_exts import GalerkinTransformer2D_Exts


def darcy_inference(cfg, checkpoint):
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
    sample_normalizer = checkpoint['Sample_Normalizer']
    target_normalizer = checkpoint['Target_Normalizer']

    # 构建模型
    torch.cuda.empty_cache()
    cfg['target_normalizer'] = target_normalizer.to(device)
    model = ModuleClz(cfg)
    model.load_state_dict(checkpoint['Module'])
    model = model.to(device)
    print(f"\nThe Numbers of Model's Parameters: {get_num_params(model)}\n")
    valid_loss_func = WeightedL2Loss2D(is_regularization=False, S=r)

    # 构建数据集
    valid_path = os.path.join(DATA_PATH, cfg['valid_dataset'])
    valid_ds = DarcyDataset(
        valid_path, num_data=cfg['num_data'], fine_resolution=R, subsample_node=sub, subsample_attn=sub_attn,
        is_training=False, noise=cfg['noise'], sample_normalizer=sample_normalizer
    )
    valid_loader = DataLoader(valid_ds, cfg['batch_size'], shuffle=False, num_workers=cfg['num_worker'], pin_memory=True)

    # 进行验证
    time_start = time.time()
    metric = validate_epoch_darcy(model, valid_loader, valid_loss_func, device=device).item()
    time_end = time.time()
    print(f'========== valid_time: {time_end - time_start:.6f} ==========')
    print(f'========== Metric : {metric} ==========')


if __name__ == '__main__':
    cfg = get_darcy_config()
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'darcy_exts_R141_R71_0624_2158.pt'), map_location='cpu')
    darcy_inference(cfg=cfg, checkpoint=checkpoint)
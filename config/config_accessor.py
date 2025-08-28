import math
import os
from ruamel.yaml import YAML

from SpGT.common.path import CONFIG_PATH
from SpGT.common.trivial import assign, get_resolution_size


def read_config(filepath):
    with open(filepath, mode='r+', encoding='UTF-8') as f:
        cfg = YAML().load(f)
    return cfg


def get_darcy_config(sub_node=None, sub_attn=None, filepath=None):
    filepath = os.path.join(CONFIG_PATH, 'darcy_config.yaml') if filepath is None else filepath
    cfg = read_config(filepath)
    if sub_node is not None:
        cfg['subsample_node'] = sub_node
        cfg['subsample_attn'] = assign(sub_attn, sub_node * 2)

    # 调整超参数配置
    R = cfg['fine_resolution']
    sub_node = cfg['subsample_node']
    sub_attn = cfg['subsample_attn']
    r_node = int((R - 1) / sub_node + 1)
    r_attn = int((R - 1) / sub_attn + 1)

    # 让 r_attn 处于 [32, 128] 范围内，并据此调整 sub_attn 的值
    r_attn = max(32, min(128, r_attn))
    sub_attn = int((R - 1) / r_attn + 1)
    cfg['subsample_attn'] = sub_attn

    # 设置降采样与升采样的值
    resolution_size = get_resolution_size(r_node, r_attn)
    cfg['resolution_size'] = resolution_size

    # 配置滤波所保留的频谱数目
    cfg['num_frequence_mode'] = math.ceil(math.sqrt(r_node))
    return cfg

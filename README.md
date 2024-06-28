# 代码组织结构

需要注意的是，项目SpGT的根目录需要位于Python解释器的搜索路径当中，假设SpGT项目位于/path/to/SpGT路径。

可以在Python解释器的site-packages目录下创建mypath.pth文件，并添加项目路径如下（一行指定一个路径）。

```
/path/to
```

也可以在所执行Python脚本的最开始位置添加如下代码。

```python
import sys
sys.path.append('/path/to')
```

```shell
SpGT
├── common
│   ├── path.py                 # 项目路径配置，所有组件都从此文件中获取路径配置
│   └── trivial.py              # 提供各种琐碎功能实现
├── storage
│   ├── data                    # 符号链接，指向数据集存储目录
│   ├── model                   # 符号链接，指向模型的存储目录
│   └── evaluation              # 评估结果的存放目录
├── config
│   ├── config_accessor.py      # 配置文件的访问器
│   └── darcy_config.yaml       # Darcy问题的配置文件
├── dataset
│   ├── data_accessor.py        # 数据文件的访问器
│   ├── darcy_dataset.py        # Darcy问题的数据集类
│   └── darcy_generate          # 生成Darcy问题所需数据的Matlab代码
├── network
│   ├── layer.py                # 神经网络模型的各种层的实现
│   ├── model.py                # 所构建的神经网络模块
│   ├── sp_layer.py             # 神经网络模型的各种层的实现，优化加速版本
│   └── sp_model.py             # 所构建的神经网络模块，优化加速版本
├── extension
│   ├── native                  # CUDA/C++层面的优化工作
│   └── bind                    # 对CUDA/C++进行封装，提供PyTorch层面的API接口
├── engine
│   ├── metric.py               # 损失函数与评估指标
│   ├── train.py                # 总体的训练过程，在给定数量的epoch上迭代
│   └── darcy_engine.py         # Darcy问题的，一轮epoch的训练或推理过程
├── run
│   ├── darcy_train.py          # Darcy模型的训练
│   ├── darcy_inference.py      # Darcy模型的推理
│   ├── ddp_darcy_train.py      # Darcy模型的训练，使用DDP框架
│   └── ddp_darcy_train.py  # Darcy模型的推理，使用DDP框架
├── evaluate                    # 针对优化工作的各种评估
└── visualize                   # 评估结果的可视化
```

# 模块与类的设计思路

配置参数由__init__()函数全部指定，但类对象仅存储需要在别处使用的配置参数。

# 命名规范

以`Sp_`前缀开始，标识各种模块的优化版本。

批量瘦长矩阵乘法的多级并行SliceK-SplitK-ReduceK策略
Multilevel parallel (SliceK-SplitK-ReduceK) strategy for batched skinny matrix multiplication

QKV矩阵与位置编码的内存布局优化，与多头层归一化融合
memory layout optimization for QKV matrices and positional encodings, multi-head layer normalization fusion

批量转置的跨步聚集和分散优化
batched transposition optimization with strided scattering and gathering
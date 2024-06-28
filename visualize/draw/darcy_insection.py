import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from SpGT.common.path import DATA_PATH, VISUALIZATION_PATH

idx = 3  # 数据索引
sample: dict = loadmat(os.path.join(DATA_PATH, 'DarcySample.mat'))
Target: np.ndarray = sample['Target'][idx].squeeze()
Input: np.ndarray = sample['Input'][idx].squeeze()
Position: np.ndarray = sample['Position'][idx].squeeze()
Grid: np.ndarray = sample['Grid'][idx].squeeze()
Output: np.ndarray = sample['Output'][idx].squeeze()
DownScaler: np.ndarray = sample['DownScaler'][idx].squeeze()
Encoder: np.ndarray = sample['Encoder'][idx].squeeze()
UpScaler: np.ndarray = sample['UpScaler'][idx].squeeze()
Decoder: np.ndarray = sample['Decoder'][idx].squeeze()
# 其中一些数据拥有多维特征，将其归约
DownScaler = np.mean(DownScaler, axis=2)
Encoder = np.mean(Encoder, axis=2)
UpScaler = np.mean(UpScaler, axis=2)
Error = np.power(Target - Output, 2)
Position = np.sum(Position, axis=2)
# 重新构建 sample
del sample
sample = dict(
    Input=Input, Output=Output, Target=Target, Error=Error,
    DownScaler=DownScaler, Encoder=Encoder, UpScaler=UpScaler, Decoder=Decoder
)

fig, axs = plt.subplots(2, 4, layout='constrained', figsize=(16, 8))
for key, ax in zip(sample.keys(), iter(axs.flatten())):
    ax.imshow(sample[key], aspect='equal')
    ax.axis('off')
plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_insection.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

# fig, ax = plt.subplots(layout='constrained', figsize=(4, 4))
# for key in sample.keys():
#     ax.imshow(sample[key], aspect='equal')
#     ax.axis('off')
#     plt.savefig(os.path.join(VISUALIZATION_PATH, f'{key}.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
#     plt.show()
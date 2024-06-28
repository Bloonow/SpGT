import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from SpGT.common.path import DATA_PATH, VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 24, 'font.weight': 'bold'})

sample: dict = loadmat(os.path.join(DATA_PATH, 'DarcySample.mat'))
Input: np.ndarray = sample['Input'].squeeze()
Target: np.ndarray = sample['Target'].squeeze()
del sample

Span = np.max(Input) - np.min(Input)
Input = Input / Span * 12
Input = Input - np.min(Input)

Input_R32   = Input[1][::16, ::16]
Target_R32  = Target[1][::16, ::16]
Input_R128  = Input[2][::4, ::4]
Target_R128 = Target[2][::4, ::4]
Input_R512  = Input[3]
Target_R512 = Target[3]

fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(16, 10))
axs[0, 0].imshow(Input_R32, aspect='equal')
axs[1, 0].imshow(Target_R32, aspect='equal')
axs[1, 0].set_xlabel('32 x 32', fontsize=32, fontweight='bold')

axs[0, 1].imshow(Input_R128, aspect='equal')
axs[1, 1].imshow(Target_R128, aspect='equal')
axs[1, 1].set_xlabel('128 x 128', fontsize=32, fontweight='bold')

b1 = axs[0, 2].imshow(Input_R512, aspect='equal')
fig.colorbar(b1)
b2 = axs[1, 2].imshow(Target_R512, aspect='equal')
fig.colorbar(b2, ticks=[0.0020, 0.0040, 0.0060, 0.0080, 0.0100, 0.0120])
axs[1, 2].set_xlabel('512 x 512', fontsize=32, fontweight='bold')

for i, j in [(i, j) for i in range(2) for j in range(3)]:
    axs[i, j].xaxis.set_major_locator(plt.NullLocator())
    axs[i, j].yaxis.set_major_locator(plt.NullLocator())

# 显示
plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_sample.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})

raw_data = [
    [   49.2,   16.8],
    [  307.3,   37.8],
    [ 1242.0,  133.8],
    [ 5156.4,  511.6],
    [21035.4, 2043.0],
]
raw_col = ['PyTorch GeMM', 'Batched Skinny GeMM']
raw_row = ['R32', 'R64', 'R128', 'R256', 'R512']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col) / 1000
raw['SpeedUp'] = raw['PyTorch GeMM'] / raw['Batched Skinny GeMM']
rawResol = raw.copy(deep=True)
print(rawResol)
raw_data = [
    [ 5073.4,   82.0],
    [ 5085.1,  142.9],
    [ 5119.0,  265.2],
    [ 5158.3,  511.2],
    [ 5212.3, 1015.7],
    [ 5377.5, 2029.8],
    [11101.9, 4059.2],
    [11395.1, 8120.0],
]
raw_col = ['PyTorch GeMM', 'Batched Skinny GeMM']
raw_row = ['B2', 'B4', 'B8', 'B16', 'B32', 'B64', 'B128', 'B256']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col) / 1000
raw['SpeedUp'] = raw['PyTorch GeMM'] / raw['Batched Skinny GeMM']
rawBatch = raw.copy(deep=True)
print(rawBatch)

# 绘图区域
fig, axs = plt.subplots(1, 2, figsize=(21, 9))
axs[0].set_title('Batch = 16')
axs[0].set_xlabel('Resolution')
axs[1].set_title('Resolution = 256')
axs[1].set_xlabel('Batch')
for ax_idx in range(2):
    ax = axs[ax_idx]
    tw = ax.twinx()
    ax.set_ylabel('Time (ms)')
    tw.set_ylabel('SpeedUp')
    df = rawResol if ax_idx == 0 else rawBatch
    # 坐标轴
    x_text = [t[1:] for t in df.index]
    x_val = np.arange(len(x_text))
    # 柱状图表示执行时间
    width = 0.35
    span = 0.4
    b1 = ax.bar(
        x=x_val, height=df['PyTorch GeMM'], width=width, label='PyTorch GeMM',
        edgecolor='black', color='royalblue'
    )
    b2 = ax.bar(
        x=x_val + span, height=df['Batched Skinny GeMM'], width=width, label='Batched Skinny GeMM',
        edgecolor='black', color='limegreen'
    )
    ax.set_xticks(x_val + span / 2, x_text)
    y_max_val = np.ceil(np.max(np.concatenate([
        df['PyTorch GeMM'], df['Batched Skinny GeMM']
    ])) / 2.5) * 2.5
    ax.set(ylim=[0, y_max_val])
    # 折线图表示加速比
    l1, = tw.plot(
        x_val + span / 2, df['SpeedUp'], label='SpeedUp',
        marker='.', markersize=16, linewidth=4, color='darkorange',
    )
    y_max_val = np.ceil(np.max(np.concatenate([
        df['SpeedUp'],
    ])) / 4) * 4
    tw.set(ylim=[0, y_max_val])

# 图例
handles = [b1, b2, l1,]
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=7)

plt.savefig(os.path.join(VISUALIZATION_PATH, 'skinny_gemm_wrt.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

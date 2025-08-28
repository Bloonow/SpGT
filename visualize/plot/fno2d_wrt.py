import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})

raw_data = [
    [ 1534.5,  1791.6,   701.5,   606.6],
    [ 1536.9,  1791.0,   702.1,   607.5],
    [ 1566.2,  3027.3,   915.2,  1559.5],
    [ 6339.7, 12961.5,  3473.6,  6302.1],
    [30378.5, 57159.0, 12688.2, 24072.0],
]
raw_col = ['FNO2D [F]', 'FNO2D [B]', 'SpFNO2D [F]', 'SpFNO2D [B]']
raw_row = ['R32', 'R64', 'R128', 'R256', 'R512']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col) / 1000
raw['SpeedUp [F]'] = raw['FNO2D [F]'] / raw['SpFNO2D [F]']
raw['SpeedUp [B]'] = raw['FNO2D [B]'] / raw['SpFNO2D [B]']
raw['SpeedUp'] = (raw['FNO2D [F]'] + raw['FNO2D [B]']) / (raw['SpFNO2D [F]'] + raw['SpFNO2D [B]'])
rawResol = raw.copy(deep=True)
print(rawResol)
raw_data = [
    [ 3.4,   6.7,  1.8,  3.2],
    [ 6.4,  13.0,  3.5,  6.3],
    [12.4,  25.4,  6.8, 12.3],
    [24.5,  50.4, 13.5, 24.8],
    [45.5, 100.4, 23.8, 49.6],
    [90.6, 200.1, 47.2, 99.1],
]
raw_col = ['FNO2D [F]', 'FNO2D [B]', 'SpFNO2D [F]', 'SpFNO2D [B]']
raw_row = ['B2', 'B4', 'B8', 'B16', 'B32', 'B64']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col)
raw['SpeedUp [F]'] = raw['FNO2D [F]'] / raw['SpFNO2D [F]']
raw['SpeedUp [B]'] = raw['FNO2D [B]'] / raw['SpFNO2D [B]']
raw['SpeedUp'] = (raw['FNO2D [F]'] + raw['FNO2D [B]']) / (raw['SpFNO2D [F]'] + raw['SpFNO2D [B]'])
rawBatch = raw.copy(deep=True)
print(rawBatch)

# 绘图区域
fig, axs = plt.subplots(1, 2, figsize=(21, 9))
axs[0].set_title('Batch = 4')
axs[0].set_xlabel('Resolution')
axs[1].set_title('Resolution = 128')
axs[1].set_xlabel('Batch')
for ax_idx in range(2):
    ax = axs[ax_idx]
    tw = ax.twinx()
    if ax_idx == 0:
        ax.set_ylabel('Time (ms)')
    if ax_idx == 1:
        tw.set_ylabel('SpeedUp')
    df = rawResol if ax_idx == 0 else rawBatch
    # 坐标轴
    x_text = [t[1:] for t in df.index]
    x_val = np.arange(len(x_text))
    # 柱状图表示执行时间
    width, span = 0.35, 0.4
    b1 = ax.bar(
        x=x_val, height=df['FNO2D [F]'], width=width, label='FNO2D [F]', bottom=np.zeros(len(x_text)),
        edgecolor='black', color='royalblue'
    )
    b2 = ax.bar(
        x=x_val, height=df['FNO2D [B]'], width=width, label='FNO2D [B]', bottom=df['FNO2D [F]'],
        edgecolor='black', color='darkorchid'
    )
    b3 = ax.bar(
        x=x_val + span, height=df['SpFNO2D [F]'], width=width, label='SpFNO2D [F]', bottom=np.zeros(len(x_text)),
        edgecolor='black', color='dodgerblue'
    )
    b4 = ax.bar(
        x=x_val + span, height=df['SpFNO2D [B]'], width=width, label='SpFNO2D [B]', bottom=df['SpFNO2D [F]'],
        edgecolor='black', color='orchid'
    )
    ax.set_xticks(x_val + span / 2, x_text)
    y_max_val = np.ceil(np.max(np.concatenate([
        df['FNO2D [F]'] + df['FNO2D [B]'], df['SpFNO2D [F]'] + df['SpFNO2D [B]']
    ])) / 50) * 50
    ax.set(ylim=[0, y_max_val])
    # 折线图表示加速比
    l1, = tw.plot(
        x_val + span / 2, df['SpeedUp [F]'], label='SpeedUp [F]',
        marker='.', markersize=16, linewidth=4, color='mediumblue',
    )
    l2, = tw.plot(
        x_val + span / 2, df['SpeedUp [B]'], label='SpeedUp [B]',
        marker='.', markersize=16, linewidth=4, color='mediumvioletred',
    )
    l3, = tw.plot(
        x_val + span / 2, df['SpeedUp'], label='SpeedUp',
        marker='.', markersize=16, linewidth=4, color='darkorange',
    )
    y_max_val = np.ceil(np.max(np.concatenate([
        df['SpeedUp [F]'], df['SpeedUp [B]'], df['SpeedUp']
    ])) / 0.5) * 0.5
    tw.set(ylim=[0, y_max_val])

# 图例
handles = [b1, b2, b3, b4, l1, l2, l3,]
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.513, 0.98), ncol=7)

plt.savefig(os.path.join(VISUALIZATION_PATH, 'fno2d_wrt.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

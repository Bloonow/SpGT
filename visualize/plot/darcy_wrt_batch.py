import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

raw_data = [
    [ 41.2,  42.1,  29.8,  28.0,  22.6,  12.5,  12.3,   8.3,   7.8,   6.4],
    [ 48.0,  47.6,  34.1,  31.7,  29.6,  16.8,  16.3,  12.2,  11.0,  10.5],
    [ 86.0,  85.7,  57.8,  56.7,  53.6,  30.4,  29.0,  21.2,  20.4,  19.5],
    [162.6, 161.6, 108.5, 107.4, 102.0,  58.5,  55.7,  39.3,  38.3,  37.7],
    [314.5, 311.9, 205.2, 205.8, 196.9, 114.9, 109.0,  75.9,  73.7,  72.6],
    [610.6, 600.7, 392.2, 402.5, 384.4, 228.9, 214.5, 146.0, 148.4, 142.6],
]
raw_col = ['Train V0', 'Train V1', 'Train V2', 'Train V3', 'Train V4', 'Valid V0', 'Valid V1', 'Valid V2', 'Valid V3', 'Valid V4']
raw_row = ['B2', 'B4', 'B8', 'B16', 'B32', 'B64']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col)
raw['GT [T]']   = raw['Train V0']
raw['SpGT [T]'] = raw['Train V4']
raw['GT [I]']   = raw['Valid V0']
raw['SpGT [I]'] = raw['Valid V4']
raw['SpeedUp [T]'] = raw['GT [T]'] / raw['SpGT [T]']
raw['SpeedUp [I]'] = raw['GT [I]'] / raw['SpGT [I]']
raw = raw.loc['B8': 'B64', 'GT [T]': 'SpeedUp [I]']
df = raw.copy(deep=True)
print(df)

# 绘图区域
fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
ax.set_axisbelow(True)
ax.grid(alpha=0.4, axis='y', linewidth=1.5)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.set_title('Resolution = 128', fontweight='bold', fontsize=26)
ax.set_xlabel('Batch Size', fontweight='bold')
tw = ax.twinx()
# ax.set_ylabel('Time (ms)', fontweight='bold')
tw.set_ylabel('SpeedUp', fontweight='bold')
# 坐标轴
x_text = [t[1:] for t in df.index]
x_val = np.arange(len(x_text))
# 柱状图表示执行时间
width, span = 0.19, 0.21
b1 = ax.bar(
    x=x_val + span * 0, height=df['GT [T]'], width=width, label='GT [T]', 
    edgecolor='black', color='royalblue'
)
b2 = ax.bar(
    x=x_val + span * 1, height=df['SpGT [T]'], width=width, label='SpGT [T]', 
    edgecolor='black', color='dodgerblue'
)
b3 = ax.bar(
    x=x_val + span * 2, height=df['GT [I]'], width=width, label='GT [I]', 
    edgecolor='black', color='darkorchid'
)
b4 = ax.bar(
    x=x_val + span * 3, height=df['SpGT [I]'], width=width, label='SpGT [I]', 
    edgecolor='black', color='orchid'
)
ax.set_xticks(x_val + span * 1.5, x_text)
y_max_val = np.ceil(np.max(np.concatenate([
    df['GT [T]'], df['SpGT [T]'], df['GT [I]'], df['SpGT [I]']
])) / 50) * 50
ax.set(ylim=[0, y_max_val])
# 折线图表示加速比
l1, = tw.plot(
    x_val + span * 1.5, df['SpeedUp [T]'], label='SpeedUp [T]',
    marker='.', markersize=18, linewidth=5, color='darkorange',
)
l2, = tw.plot(
    x_val + span * 1.5, df['SpeedUp [I]'], label='SpeedUp [I]',
    marker='.', markersize=18, linewidth=5, color='green',
)
y_max_val = np.ceil(np.max(np.concatenate([
    df['SpeedUp [T]'], df['SpeedUp [I]']
])) / 0.1) * 0.1
tw.set(ylim=[1.0, y_max_val])

# 图例
handles = [b1, b2, b3, b4, l1, l2,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='lower left', fontsize=20, bbox_to_anchor=(0., 0.25))

plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_wrt_batch.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

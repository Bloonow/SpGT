import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

raw_data = [
    [ 41.3,  42.3,  29.7,  28.4,  22.7,  12.4,  12.3,   8.4,   7.9,   6.6],
    [ 40.1,  41.8,  29.5,  28.1,  22.5,  12.5,  12.5,   8.5,   8.0,   6.4],
    [ 47.4,  47.4,  34.1,  31.7,  29.7,  16.7,  16.2,  12.2,  10.9,  10.5],
    [174.8, 173.2, 121.0, 111.0, 104.6,  63.8,  60.6,  45.0,  39.7,  37.6],
    [675.6, 677.4, 482.9, 427.1, 394.9, 256.2, 241.9, 178.5, 153.0, 147.6],
]
raw_col = ['Train V0', 'Train V1', 'Train V2', 'Train V3', 'Train V4', 'Valid V0', 'Valid V1', 'Valid V2', 'Valid V3', 'Valid V4']
raw_row = ['R32', 'R64', 'R128', 'R256', 'R512']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col)
raw['GT [T]']   = raw['Train V0']
raw['SpGT [T]'] = raw['Train V4']
raw['GT [I]']   = raw['Valid V0']
raw['SpGT [I]'] = raw['Valid V4']
raw['SpeedUp [T]'] = raw['GT [T]'] / raw['SpGT [T]']
raw['SpeedUp [I]'] = raw['GT [I]'] / raw['SpGT [I]']
raw = raw.loc['R64': 'R512', 'GT [T]': 'SpeedUp [I]']
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
ax.set_title('Batch Size = 4', fontweight='bold', fontsize=26)
ax.set_xlabel('Resolution', fontweight='bold')
ax.set_ylabel('Time (ms)', fontweight='bold')
tw = ax.twinx()
# tw.set_ylabel('SpeedUp')
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

plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_wrt_resol.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

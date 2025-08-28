import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

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
raw = raw.loc['B4': 'B64']
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
# ax.set_ylabel('Time (ms)', fontweight='bold')
tw = ax.twinx()
tw.set_ylabel('SpeedUp', fontweight='bold')
# 坐标轴
x_text = [t[1:] for t in df.index]
x_val = np.arange(len(x_text))
# 柱状图表示执行时间
width, span = 0.36, 0.4
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
    marker='.', markersize=18, linewidth=5, color='mediumblue',
)
l2, = tw.plot(
    x_val + span / 2, df['SpeedUp [B]'], label='SpeedUp [B]',
    marker='.', markersize=18, linewidth=5, color='mediumvioletred',
)
l3, = tw.plot(
    x_val + span / 2, df['SpeedUp'], label='SpeedUp',
    marker='.', markersize=18, linewidth=5, color='darkorange',
)
y_max_val = np.ceil(np.max(np.concatenate([
    df['SpeedUp [F]'], df['SpeedUp [B]'], df['SpeedUp']
])) / 0.2) * 0.2
tw.set(ylim=[1., y_max_val])

# 图例
handles = [b1, b2, b3, b4, l1, l2, l3,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='lower left', fontsize=20, bbox_to_anchor=(0., 0.125))

plt.savefig(os.path.join(VISUALIZATION_PATH, 'fno2d_wrt_batch.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

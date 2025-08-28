import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

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
# tw.set_ylabel('SpeedUp', fontweight='bold')
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
])) / 0.5) * 0.5
tw.set(ylim=[1., y_max_val])
tw.set_yticks([1., 1.4, 1.8, 2.2, 2.6, 3.0])

# 图例
handles = [b1, b2, b3, b4, l1, l2, l3,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='lower left', fontsize=20, bbox_to_anchor=(0., 0.125))

plt.savefig(os.path.join(VISUALIZATION_PATH, 'fno2d_wrt_resol.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

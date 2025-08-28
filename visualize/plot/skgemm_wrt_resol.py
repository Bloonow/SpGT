import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

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
ax.set_title('Batch Size = 16', fontweight='bold', fontsize=26)
ax.set_xlabel('Resolution', fontweight='bold')
ax.set_ylabel('Time (ms)', fontweight='bold')
tw = ax.twinx()
# tw.set_ylabel('SpeedUp')
# 坐标轴
x_text = [t[1:] for t in df.index]
x_val = np.arange(len(x_text))
# 柱状图表示执行时间
width = 0.36
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
])) / 5) * 5
ax.set(ylim=[0, y_max_val])
# 折线图表示加速比
l1, = tw.plot(
    x_val + span / 2, df['SpeedUp'], label='SpeedUp',
    marker='.', markersize=18, linewidth=5, color='darkorange',
)
y_max_val = np.ceil(np.max(np.concatenate([
    df['SpeedUp'],
])) / 4) * 4
tw.set(ylim=[.5, y_max_val])

# 图例
handles = [b1, b2, l1,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='upper left', fontsize=20, bbox_to_anchor=(0., 0.96))

plt.savefig(os.path.join(VISUALIZATION_PATH, 'skinny_gemm_wrt_resol.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

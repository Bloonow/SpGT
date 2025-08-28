import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

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
ax.set_title('Resolution = 256', fontweight='bold', fontsize=26)
ax.set_xlabel('Batch Size', fontweight='bold')
# ax.set_ylabel('Time (ms)', fontweight='bold')
tw = ax.twinx()
tw.set_ylabel('SpeedUp', fontweight='bold')
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
ax.set_yticks([0, 3, 6, 9, 12, 15])
# 折线图表示加速比
l1, = tw.plot(
    x_val + span / 2, df['SpeedUp'], label='SpeedUp',
    marker='.', markersize=18, linewidth=5, color='darkorange',
)
tw.set_yticks([1, 10, 20, 30, 40, 50, 60])
y_max_val = np.ceil(np.max(np.concatenate([
    df['SpeedUp'],
])) / 4) * 4
tw.set(ylim=[0, y_max_val])

# 图例
handles = [b1, b2, l1,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='upper left', fontsize=20, bbox_to_anchor=(0., 0.96))

plt.savefig(os.path.join(VISUALIZATION_PATH, 'skinny_gemm_wrt_batch.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

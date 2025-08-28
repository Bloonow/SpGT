import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

raw_data = [
    [4998.620203, 2633.813235, 1800.568480, 1381.715324, 1164.057796, 1015.852081,  966.600997,  874.296220],
    [5291.233654, 5481.692706, 5633.015934, 5650.536486, 5752.888575, 5810.238053, 5718.157950, 5840.771904],
]
raw_col = ['GPU1', 'GPU2', 'GPU3', 'GPU4', 'GPU5', 'GPU6', 'GPU7', 'GPU8']
raw_row = ['Time [S]', 'Time [W]']
df = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col)
for gpu in raw_col:
    df.loc['SpeedUp [S]', gpu] = df.loc['Time [S]', raw_col[0]] / df.loc['Time [S]', gpu]
    df.loc['SpeedUp [W]', gpu] = (df.loc['Time [W]', raw_col[0]] * int(gpu[3:])) / df.loc['Time [W]', gpu]
    df.loc['Efficiency [S]', gpu] = df.loc['Time [S]', raw_col[0]] / (df.loc['Time [S]', gpu] * int(gpu[3:]))
    df.loc['Efficiency [W]', gpu] = df.loc['Time [W]', raw_col[0]] / df.loc['Time [W]', gpu]
print(df)

# 绘图区域
fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
ax.set_axisbelow(True)
ax.grid(alpha=0.4, axis='y', linewidth=1.5)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
tw = ax.twinx()
ax.set_xlabel('GPUs', fontweight='bold')
ax.set_ylabel('SpeedUp', fontweight='bold')
tw.set_ylabel('Efficiency', fontweight='bold')
# 坐标轴
x_text = [t[3:] for t in df.columns]
x_val = np.arange(len(x_text))
y_val = np.linspace(0, 1, 11)
y_text = [f'{v * 100:.0f}%' for v in y_val]
# 折线图表示加速比
l1, = ax.plot(
    x_val, df.loc['SpeedUp [S]'], label='SpeedUp [S]',
    linewidth=3, markersize=10, marker='x',
    color='blue',
)
l2, = ax.plot(
    x_val, df.loc['SpeedUp [W]'], label='SpeedUp [W]',
    linewidth=3, markersize=10, marker='+',
    color='green',
)
ax.set_xticks(x_val, x_text)
ax.set(ylim=[0.5, 8.5])
# 折线图表示并行效率
l3, = tw.plot(
    x_val, df.loc['Efficiency [S]'], label='Efficiency [S]',
    linewidth=3, markersize=10, linestyle=(0, (2, 1)), marker='^',
    color='darkorange',
)
l4, = tw.plot(
    x_val, df.loc['Efficiency [W]'], label='Efficiency [W]',
    linewidth=3, markersize=10, linestyle=(0, (2, 1)), marker='v',
    color='mediumvioletred',
)
tw.set_yticks(y_val, y_text)
tw.set(ylim=[0.595, 1.025])

handles = [l1, l2, l3, l4,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=12)

plt.savefig(os.path.join(VISUALIZATION_PATH, 'ddp_efficiency.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()
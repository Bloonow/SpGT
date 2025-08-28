import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

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
for R in raw_row:
    raw.loc[f'SpeedUp_{R}', 'Train V0': 'Train V4'] = raw.loc[f'{R}', 'Train V0'] / raw.loc[f'{R}', 'Train V0': 'Train V4']
    raw.loc[f'SpeedUp_{R}', 'Valid V0': 'Valid V4'] = raw.loc[f'{R}', 'Valid V0'] / raw.loc[f'{R}', 'Valid V0': 'Valid V4']
    raw.loc[f'SpeedUp_{R}'] = np.round(raw.loc[f'SpeedUp_{R}'], decimals=2)
print(raw)

# 绘图区域
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_axisbelow(True)
ax.grid(alpha=0.4, axis='x', linewidth=1.5)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.set_title('[Inference] Batch = 4, Resolution = 512', fontweight='bold')
ax.set_xlabel('SpeedUp', fontweight='bold')
# 坐标轴
y_text = ['Baseline', '+ProjPos', '+Lnorm', '+SkGeMM', '+FNO2D']
y_val = np.arange(len(y_text))
height = 0.5
b1 = ax.barh(
    y=y_val, width=raw.loc['SpeedUp_R512', 'Valid V0': 'Valid V4'], height=height,
    edgecolor='black', color=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
)
ax.bar_label(b1, padding=3, labels=raw.loc['SpeedUp_R512', 'Valid V0': 'Valid V4'])
ax.set_yticks(y_val, y_text)
ax.invert_yaxis()  # Y轴从上到下
x_max_val = np.ceil(np.max(np.concatenate([
    raw.loc['SpeedUp_R512', 'Valid V0': 'Valid V4']
])) / 0.4) * 0.4
ax.set(xlim=[.9, 1.9])

# 显示
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_ablation.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

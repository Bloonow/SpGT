import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

raw_data = [
    [  176.4,   182.3,    60.0,   185.4,    439.0,   1092.2,   102.4,   301.9],
    [  644.2,   552.4,   199.6,   564.7,   1762.0,   4162.3,   391.8,  1059.0],
    [ 2738.6,  1927.9,   739.6,  1970.5,   7128.6,  16066.2,  1410.3,  3873.7],
    [12242.5,  7158.9,  2878.4,  7684.7,  29096.0,  64167.5,  5413.1, 15457.2],
    [40690.8, 29142.8, 11400.5, 29863.0, 106481.2, 257244.1, 21382.6, 61003.0],
]
raw_col = [
    'ProjPos [F]', 'ProjPos [B]', 'SpProjPos [F]', 'SpProjPos [B]',
    'ProjPos_Lnorm [F]', 'ProjPos_Lnorm [B]', 'SpProjPos_Lnorm [F]', 'SpProjPos_Lnorm [B]'
]
raw_row = ['R32', 'R64', 'R128', 'R256', 'R512']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col) / 1000
raw['ProjPos']         = raw['ProjPos [F]']         + raw['ProjPos [B]']
raw['SpProjPos']       = raw['SpProjPos [F]']       + raw['SpProjPos [B]']
raw['ProjPos_Lnorm']   = raw['ProjPos_Lnorm [F]']   + raw['ProjPos_Lnorm [B]']
raw['SpProjPos_Lnorm'] = raw['SpProjPos_Lnorm [F]'] + raw['SpProjPos_Lnorm [B]']
raw['Lnorm']   = raw['ProjPos_Lnorm']   - raw['ProjPos']
raw['SpLnorm'] = raw['SpProjPos_Lnorm'] - raw['SpProjPos']
raw['SpeedUp ProjPos'] = raw['ProjPos']       / raw['SpProjPos']
raw['SpeedUp Lnorm']   = raw['Lnorm']         / raw['SpLnorm']
raw['SpeedUp']         = raw['ProjPos_Lnorm'] / raw['SpProjPos_Lnorm']
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
    x=x_val, height=df['ProjPos'], width=width, label='ProjPos', bottom=np.zeros(len(x_text)),
    edgecolor='black', color='royalblue'
)
b2 = ax.bar(
    x=x_val, height=df['Lnorm'], width=width, label='Lnorm', bottom=df['ProjPos'],
    edgecolor='black', color='darkorchid'
)
b3 = ax.bar(
    x=x_val + span, height=df['SpProjPos'], width=width, label='SpProjPos', bottom=np.zeros(len(x_text)),
    edgecolor='black', color='dodgerblue'
)
b4 = ax.bar(
    x=x_val + span, height=df['SpLnorm'], width=width, label='SpLnorm', bottom=df['SpProjPos'],
    edgecolor='black', color='orchid'
)
ax.set_xticks(x_val + span / 2, x_text)
y_max_val = np.ceil(np.max(np.concatenate([
    df['ProjPos'] + df['Lnorm'], df['SpProjPos'] + df['SpLnorm']
])) / 50) * 50
ax.set(ylim=[0, y_max_val])
# 折线图表示加速比
l1, = tw.plot(
    x_val + span / 2, df['SpeedUp ProjPos'], label='SpeedUp ProjPos',
    marker='.', markersize=18, linewidth=5, color='mediumblue',
)
l2, = tw.plot(
    x_val + span / 2, df['SpeedUp Lnorm'], label='SpeedUp Lnorm',
    marker='.', markersize=18, linewidth=5, color='mediumvioletred',
)
l3, = tw.plot(
    x_val + span / 2, df['SpeedUp'], label='SpeedUp',
    marker='.', markersize=18, linewidth=5, color='darkorange',
)
y_max_val = np.ceil(np.max(np.concatenate([
    df['SpeedUp ProjPos'], df['SpeedUp Lnorm'], df['SpeedUp']
])) / 0.5) * 0.5
tw.set(ylim=[1., y_max_val])

# 图例
handles = [b1, b2, b3, b4, l1, l2, l3,]
labels = [h.get_label() for h in handles]
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(0., 0.7), fontsize=18)

plt.savefig(os.path.join(VISUALIZATION_PATH, 'projpos_lnorm_wrt_resol.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

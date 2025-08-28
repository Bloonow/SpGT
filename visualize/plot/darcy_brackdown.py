import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpGT.common.path import VISUALIZATION_PATH

# 设置全局字体大小
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

raw_data = [
    [  684.9,   592.7,   188.5,   199.0,   9716.1,   7757.5,  4009.3,   4296.6,  3766.9,  3847.2,  1742.1,  1392.4],
    [  676.2,   580.0,   217.4,   235.8,   9914.5,   7898.7,  4011.9,   4298.0,  3697.0,  3964.2,  1760.6,  1437.0],
    [ 1266.2,  1513.2,   832.9,   881.6,  12170.4,  13170.9,  5924.7,   9768.5,  3696.2,  4007.2,  2055.9,  1500.3],
    [ 5027.1,  6048.6,  3204.6,  3398.7,  44660.9,  46133.7, 20549.3,  33985.0,  9361.6, 10710.1,  7928.4,  5550.7],
    [21164.4, 24947.1, 12838.4, 13589.6, 182762.8, 177385.6, 81711.3, 129317.3, 38912.1, 45332.4, 30885.6, 21299.7],
]
raw_col = [
    'DownScaler [F]', 'DownScaler [B]', 'UpScaler [F]', 'UpScaler [B]',
    'EncoderLayer [F]', 'EncoderLayer [B]', 'SpEncoderLayer [F]', 'SpEncoderLayer [B]',
    'SpectralRegressor [F]', 'SpectralRegressor [B]', 'SpSpectralRegressor [F]', 'SpSpectralRegressor [B]'
]
raw_row = ['R32', 'R64', 'R128', 'R256', 'R512']
raw = pd.DataFrame(data=raw_data, index=raw_row, columns=raw_col) / 1000
raw['DownScaler']          = raw['DownScaler [F]']          + raw['DownScaler [B]']
raw['UpScaler']            = raw['UpScaler [F]']            + raw['UpScaler [B]']
raw['EncoderLayer']        = raw['EncoderLayer [F]']        + raw['EncoderLayer [B]']
raw['SpEncoderLayer']      = raw['SpEncoderLayer [F]']      + raw['SpEncoderLayer [B]']
raw['SpectralRegressor']   = raw['SpectralRegressor [F]']   + raw['SpectralRegressor [B]']
raw['SpSpectralRegressor'] = raw['SpSpectralRegressor [F]'] + raw['SpSpectralRegressor [B]']
raw = raw.loc['R512']

row = ['DownScaler', 'EncoderLayer', 'UpScaler', 'SpectralRegressor']
col = ['GT [T]', 'SpGT [T]', 'GT [I]', 'SpGT [I]']
df = pd.DataFrame(index=row, columns=col)
df.loc['DownScaler']        = [raw['DownScaler'],        raw['DownScaler'],          raw['DownScaler [F]'],        raw['DownScaler [F]']]
df.loc['EncoderLayer']      = [raw['EncoderLayer'],      raw['SpEncoderLayer'],      raw['EncoderLayer [F]'],      raw['SpEncoderLayer [F]']]
df.loc['UpScaler']          = [raw['UpScaler'],          raw['UpScaler'],            raw['UpScaler [F]'],          raw['UpScaler [F]']]
df.loc['SpectralRegressor'] = [raw['SpectralRegressor'], raw['SpSpectralRegressor'], raw['SpectralRegressor [F]'], raw['SpSpectralRegressor [F]']]
# 因为模型中 EncoderLayer 为 6 层，SpectralRegressor 为 2 层，此处取特定层数的时间
df.loc['EncoderLayer']      /= 1
df.loc['SpectralRegressor'] /= 1
df.loc['Model'] = df.loc['DownScaler'] + df.loc['EncoderLayer'] + df.loc['UpScaler'] + df.loc['SpectralRegressor']
df.loc['DownScaler [%]']        = df.loc['DownScaler']        #/ df.loc['Model'] * 100
df.loc['EncoderLayer [%]']      = df.loc['EncoderLayer']      #/ df.loc['Model'] * 100
df.loc['UpScaler [%]']          = df.loc['UpScaler']          #/ df.loc['Model'] * 100
df.loc['SpectralRegressor [%]'] = df.loc['SpectralRegressor'] #/ df.loc['Model'] * 100
print(df)

# 坐标轴
x_text = ['GT [T]', 'SpGT [T]', 'GT [I]', 'SpGT [I]']
x_val = np.arange(len(x_text))

# 绘图区域
fig, ax = plt.subplots(figsize=(9, 8))
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.set_title('Batch = 4, Resolution = 512', fontweight='bold')
ax.set_ylabel('Time (ms)', fontweight='bold')
width = 0.57
bottom = np.zeros(len(x_text))
b1 = ax.bar(
    x=x_val, height=df.loc['DownScaler [%]'], width=width, bottom=bottom, label='DownSacler', 
    edgecolor='black', color='tab:blue', hatch=r'--'
)
bottom += df.loc['DownScaler [%]']
b2 = ax.bar(
    x=x_val, height=df.loc['EncoderLayer [%]'], width=width, bottom=bottom, label='EncoderLayer', 
    edgecolor='black', color='tab:orange', hatch=r'||'
)
bottom += df.loc['EncoderLayer [%]']
b3 = ax.bar(
    x=x_val, height=df.loc['UpScaler [%]'] , width=width, bottom=bottom, label='UpScaler', 
    edgecolor='black', color='tab:green', hatch=r'xxx'
)
bottom += df.loc['UpScaler [%]'] 
b4 = ax.bar(
    x=x_val, height=df.loc['SpectralRegressor [%]'], width=width, bottom=bottom, label='DecoderLayer', 
    edgecolor='black', color='tab:purple', hatch=r'++'
)
ax.set_xticks(x_val, x_text)
ax.set_ylim(0, df.loc['Model'].max())
# ax.set_ylim(0, 100)
ax.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, 'darcy_breakdown.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

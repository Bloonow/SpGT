import os
import matplotlib.pyplot as plt
import numpy as np

from SpGT.common.path import VISUALIZATION_PATH

num = 3
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)
x_list = [i * 2 * x + i * np.pi / 2 for i in range(1, num + 1)]
y_list = [np.sin(x) for x in x_list]

fig, axs = plt.subplots(num, 1)
for i in range(num):
    ax = axs[i]
    x, y = x_list[i], y_list[i]
    ax.plot(x, y, linewidth=5, color='red')
    ax.axis('off')

plt.savefig(os.path.join(VISUALIZATION_PATH, 'sin.png'), format='png', transparent=True, dpi=360, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

root = Path('data/out_nve')
out = Path('figures')
L = 10


if not out.is_dir():
    out.mkdir()

df = pd.read_csv(root / 'out_0.csv', index_col=0)
df1000 = pd.read_csv(root / 'out_2000.csv', index_col=0)

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

ax1.scatter(df.x, df.y, df.z)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

ax1.set_xlim([0, L])
ax1.set_ylim([0, L])
ax1.set_zlim([0, L])

ax1.set_title('a)', fontsize=24)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax2.scatter(df1000.x, df1000.y, df1000.z)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

ax2.set_xlim([0, L])
ax2.set_ylim([0, L])
ax2.set_zlim([0, L])

ax2.set_title('b)', fontsize=24)

plt.subplots_adjust(hspace=0.1)

plt.tight_layout()

fig.savefig(out / 'initial_final_conditions.pdf')
plt.show()

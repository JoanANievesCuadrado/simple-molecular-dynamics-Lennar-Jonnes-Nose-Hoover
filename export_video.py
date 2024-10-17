import matplotlib.pyplot as plt
import pandas as pd
import re
import subprocess

from pathlib import Path
from tqdm import tqdm

root = Path('out')
out = root / 'images'

for i, fname in enumerate(tqdm(root.iterdir())):
    match = re.match(r'out_(\d+).csv', fname.name)
    if not match:
        continue

    n = match.group(1)
    df = pd.read_csv(fname, index_col=0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df.x, df.y, df.z)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_xlim([0, 7.5])
    ax.set_ylim([0, 7.5])
    ax.set_zlim([0, 7.5])
    
    fig.savefig(out / f'frame_{n}.png')
    plt.close()    
    
subprocess.run('ffmpeg -framerate 90 -i out/images/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4', shell=True)

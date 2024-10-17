import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path


sns.set_style('darkgrid') 
thermo_file = Path('data/thermo_nve.txt')
thermostat_file = Path('data/thermostat_nve.txt')
FS = 12


thermo  = pd.read_table(thermo_file, names=['et', 'ek', 'v', 't', 'p'])


N = thermo.shape[0]

em = thermo.et.iloc[-100:].mean() * np.ones(N)
tm = thermo.t.iloc[-100:].mean() * np.ones(N)
pm = thermo.p.iloc[-100:].mean() * np.ones(N)
x = 10 * np.arange(N)

figt, axt = plt.subplots()

color2 = 'tab:orange'
color3 = 'tab:green'
axt.plot(x, thermo.t, label='T', color=color2)
axt.plot(x, tm, '--', color=color2, label=r'$\left<T\right>$' f' = {tm[0]:.2f}')
axt.plot(x, thermo.p, label='P', color=color3)
axt.plot(x, pm, '--', color=color3, label=r'$\left<P\right>$' f' = {pm[0]:.2f}')

axt.set_ylabel('T, P', fontsize=FS)
axt.set_xlabel('Iteraciones', fontsize=FS)

axe = axt.twinx()
color1 = 'tab:blue'
axe.plot(x, thermo.et, label='$E_T$', color=color1)
axe.plot(x, em, '--', color=color1, label=r'$\left<E_T\right>$' f' = {em[0]:.2f}')
axe.set_ylabel('E', fontsize=FS, color=color1)

axe.set_ylim([-1176.5 , -1128.5])

plt.tight_layout()
axt.tick_params(axis='both', which='major', labelsize=FS)
axe.tick_params(axis='both', which='major', labelsize=FS)
axe.tick_params(axis='y', which='major', colors=color1)

# Combina las leyendas
lines, labels = axt.get_legend_handles_labels()
lines2, labels2 = axe.get_legend_handles_labels()
axe.legend(lines + lines2, labels + labels2, loc='upper right')

figt.savefig('figures/nve_T_and_P_and_E.pdf', bbox_inches='tight')

plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path


sns.set_style('darkgrid')
thermo_file = Path('data/thermo_nvt_progress.txt')
thermostat_file = Path('data/thermostat_nvt_progress.txt')
FS = 14

thermo  = pd.read_table(thermo_file, names=['et', 'ek', 'v', 't', 'p'])
thermostat = np.loadtxt(thermostat_file)

N = thermo.shape[0]

em = thermo.et.iloc[-2000:].mean() * np.ones(N)
tm = thermo.t.iloc[-2000:].mean() * np.ones(N)
pm = thermo.p.iloc[-2000:].mean() * np.ones(N)

x = 10 * np.arange(thermo.shape[0])

color1 = 'tab:blue'
color2 = 'tab:orange'
color3 = 'tab:green'

figt, axt = plt.subplots()

axt.plot(x[::10], thermo.t[::10], label='T', color=color2)
axt.plot(x[::10], thermo.p[::10], label='P', color=color3)
axt.plot(x[::10], tm[::10], '--', color=color2, label=r'$\left<T\right>$' f' = {tm[0]:.2f}')
axt.plot(x[::10], pm[::10], '--', color=color3, label=r'$\left<P\right>$' f' = {pm[0]:.2f}')

axt.set_ylabel('T, P', fontsize=FS)
axt.set_xlabel('Iteraciones', fontsize=FS)

axe = axt.twinx()
axe.plot(x[::10], thermo.et[::10], label='$E_T$', color=color1)
axe.plot(x[::10], em[::10], '--', color=color1, label=r'$\left<E_T\right>$' f' = {em[0]:.2f}')
axe.set_ylabel('E', fontsize=FS, color=color1)

axt.set_ylim([0.6817, 15.376])
axe.set_ylim([-2415-125 , -578-125])

plt.tight_layout()
axt.tick_params(axis='both', which='major', labelsize=FS)
axe.tick_params(axis='both', which='major', labelsize=FS)
axe.tick_params(axis='y', which='major', colors=color1)

# Combina las leyendas
lines, labels = axt.get_legend_handles_labels()
lines2, labels2 = axe.get_legend_handles_labels()

lines_ = [lines[0], lines[1], lines2[0], lines[2], lines[3], lines2[1]]
labels_ = [labels[0], labels[1], labels2[0], labels[2], labels[3], labels2[1]]
axe.legend(lines_, labels_, loc='lower left', ncol=2, bbox_to_anchor=(0.0, 0.1))

# axt.legend()
figt.savefig('figures/nvt_progress_T_and_P_and_E.pdf', bbox_inches='tight')

#===============================================================================

fig, ax = plt.subplots()

xim = thermostat[-2000:].mean() * np.ones(thermostat.shape[0])

color = 'tab:blue'

ax.plot(x[::10], thermostat[::10], label=r'$\dot{\xi}$', color=color)
ax.plot(x[::10], xim[::10], '--', label=r'$\langle\dot{\xi}\rangle$ = ' f'{xim[0]:.2f}', color=color)

ax.set_xlabel('Iteraciones', fontsize=FS)
ax.set_ylabel(r'$\dot{\xi}$', fontsize=FS)
ax.tick_params(axis='both', which='major', labelsize=FS)

ax.set_ylim([-0.05, 0.05])

plt.tight_layout()
plt.legend()

fig.savefig('figures/nvt_progress_xi.pdf')

plt.show()

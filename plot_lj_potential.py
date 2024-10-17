import matplotlib.pyplot as plt
import numpy as np

def lj(x, s=1, e=1):
    return 4*e*(np.power(s/x, 12) - np.power(s/x, 6))


def lj2(x, r_cut=2.5, s=1, e=1):
    v_cut = 4 * (np.power(r_cut, -12) - np.power(r_cut, -6))
    f_cut = 24 * (2 * np.power(r_cut, -13) - np.power(r_cut, -7))
    p = lj(x, s, e)  - v_cut + (x - r_cut) * f_cut

    p[x >= r_cut] = 0.0
    return p

# Generate some sample data
x = np.linspace(0.9, 2.7, 100)
y = lj(x)
y2 = lj2(x, 1.5)

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y, 'r', label='LJ')
ax.plot(x, y2, 'b', label='LJ truncado y desplazado')


ax.plot([0, 2**(1/6)], [-1, -1], 'k--')
ax.plot([2**(1/6), 2**(1/6)], [0, -1], 'k--')

ax.plot([0, 1.12315], [-0.929973, -0.929973], 'k--')
ax.plot([1.12315, 1.12351], [0, -0.929973], 'k--')

# ax.text(2**(1/6), 0.15, '$\sqrt[6]{2}\sigma$', ha='center', va='top')
# ax.text(-0.1, -1, '$-\epsilon$', ha='right', va='center')

ax.set_xticks([2**(1/6), 2.5, 1.12351], labels=['$r_{min}$', '$r_{cut}$', "\n\n$r'_{min}$"], rotation=30)
ax.set_yticks([-1,-0.929973], labels=[r'$-\epsilon_0$', r"$-\epsilon_0'$"])

ax.set_xlabel('r', loc='right', labelpad=-150)
ax.set_ylabel('$U_{LJ}(r)$', loc='top', rotation=0, labelpad=-20)

# Move the x-axis to y=0
ax.spines['top'].set_position(('data', 0))
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

# Hide the top and right spines
ax.spines['bottom'].set_color('none')
ax.spines['right'].set_color('none')

ax.set_ylim([-1.2, 1.2])
ax.set_xlim((0.945, 2.7))

plt.legend()
plt.xticks(rotation=45)

plt.savefig('figures/lj_potential.pdf', bbox_inches='tight')

# Show the plot
plt.show()
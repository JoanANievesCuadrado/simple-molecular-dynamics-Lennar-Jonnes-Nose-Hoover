import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Generar puntos aleatorios
np.random.seed(0)
points = np.random.rand(100, 2)

# Punto central y radio de la circunferencia
center = points[0]
radius = 0.2

# Calcular distancias y encontrar puntos dentro del radio
distances = np.linalg.norm(points - center, axis=1)
inside_circle = distances < radius
inside_circle[0] = False

# Crear el plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.1)

# Primer gráfico: circunferencia y puntos dentro
ax1.scatter(points[:, 0], points[:, 1], color='gray', label='Puntos')
ax1.scatter(center[0], center[1], color='red', label='Punto central')
circle = plt.Circle(center, radius, color='blue', fill=False, linestyle='--', label='Circunferencia')
ax1.add_patch(circle)
ax1.scatter(points[inside_circle, 0], points[inside_circle, 1], color='green', label='Dentro de la circunferencia')

# Añadir la flecha inclinada 30 grados
angle = np.radians(30)
radius -= 0.02
dx = radius * np.cos(angle)
dy = radius * np.sin(angle)
ax1.arrow(center[0], center[1], dx, dy, head_width=0.02, head_length=0.02, fc='black', ec='black')
ax1.text(center[0] + dx/2 +0.01, center[1] + dy/2 - 0.03, r'$r_{\text{cut}}$', fontsize=12)

# Segundo gráfico: celdas y puntos vecinos
ax2.scatter(points[:, 0], points[:, 1], color='gray', label='Puntos')


# Dividir el espacio en celdas
cell_size = 0.2
grid_x = np.floor(points[:, 0] / cell_size).astype(int)
grid_y = np.floor(points[:, 1] / cell_size).astype(int)
center_cell = (grid_x[0], grid_y[0])

x_cells = np.arange(0, 1 + cell_size, cell_size)
y_cells = np.arange(0, 1 + cell_size, cell_size)


# Dibujar las celdas
for x in x_cells:
    for y in y_cells:
        rect = Rectangle((x, y), cell_size, cell_size, fill=None, edgecolor='gray', linestyle='--', alpha=0.3)
        ax2.add_patch(rect)

# Dibujar celdas
for i in range(-1, 2):
    for j in range(-1, 2):
        cell_rect = plt.Rectangle(((center_cell[0] + i)* cell_size, (center_cell[1] + j)* cell_size) , cell_size, cell_size, edgecolor='blue', fill=False, linestyle='--', linewidth=1.5)
        ax2.add_patch(cell_rect)

mx = (grid_x[0] - 1) * cell_size
Mx = (grid_x[0] + 1) * cell_size
my = (grid_y[0] - 1) * cell_size
My = (grid_y[0] + 1) * cell_size

# Encontrar celdas vecinas
neighbor_cells = []
for x in x_cells:
    for y in y_cells:
        if (mx <= x <= Mx) and (my <= y <= My) and x != center[0] and y != center[1]:
            neighbor_cells.append((x, y))

# Marcar puntos en celdas vecinas
for cell in neighbor_cells:
    x, y = cell
    in_cell = (points[:, 0] >= x) & (points[:, 0] < x + cell_size) & (points[:, 1] >= y) & (points[:, 1] < y + cell_size)
    ax2.scatter(points[in_cell, 0], points[in_cell, 1], color='green')

ax2.scatter(center[0], center[1], color='red', label='Punto central')

ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xticks(np.arange(0,1.1,0.2), labels=[])
ax1.set_yticks(np.arange(0,1.1,0.2), labels=[])
ax2.set_xticks(np.arange(0,1.1,0.2), labels=[])
ax2.set_yticks(np.arange(0,1.1,0.2), labels=[])

ax1.text(0.02, 0.98, 'a)', transform=ax1.transAxes, fontsize=25, verticalalignment='top')
ax2.text(0.02, 0.98, 'b)', transform=ax2.transAxes, fontsize=25, verticalalignment='top')

plt.savefig('figures/verlet_and_cells_list.pdf', bbox_inches='tight')

plt.show()

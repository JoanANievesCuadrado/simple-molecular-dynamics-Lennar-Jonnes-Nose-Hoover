## System vars
## ===========

sigma = 1.0     # distance scale
e0 = 1.0        # minimum of the potential energy
kb = 1.0        # boltzmann constant
m = 1.0         # mass of the particles
r_cut = 2.5     # cutoff of the potential energy
STEPS = 100000    # Number of iterations
dt = 0.001      # timestep
rho = 1       # Reduced density
SEED = 4873955  # Random number generator seed
T0 = 1.0        # Expectate initial temperature
T_end = 1.5     # Temperature of the thermostat
xlo, xhi, ylo, yhi, zlo, zhi = 0, 10, 0, 10, 0, 10
Q = 1      # Nose-hoover thermostate parameter
# El valor de Q es importante porque para valores pequeños la temperatura
# oscila y demora mucho en estabilizar, y para valores muy grandes el sistema
# demora mucho en alcanzar el valor del termostato

save_n = 10      # Number od steps to save the state of the system
thermo_n = 10    # Number of steps to compute and sace de thermodynamics parameters


## Import usefull packages
## =======================

import logging
import numpy as np
import pandas as pd
import re

from logging.handlers import RotatingFileHandler
from pathlib import Path
from tqdm import tqdm
from typing import Generator, Tuple


# Logger configuration 
# ====================

rng = np.random.default_rng(SEED)
ROOT = Path('./' )
OUT_PATH = ROOT / Path('out')
IMAGES_PATH =  OUT_PATH / 'images'
THERMO_FILE = ROOT / 'thermo.txt'
THERMOSTAT_FILE = ROOT / 'thermostat.txt'

if not OUT_PATH.is_dir():
    OUT_PATH.mkdir()

if not IMAGES_PATH.is_dir():
    IMAGES_PATH.mkdir()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_format = logging.Formatter('[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s')
console_handler.setFormatter(console_format)

file_handler = RotatingFileHandler('md.log', maxBytes=10 * 1024 * 1024, backupCount=10)
file_format = logging.Formatter('[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s')
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.WARNING)

logger = logging.getLogger('MD')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# logger.setLevel(logging.INFO)

# Custom log level for thermo_log
THERMO_LEVEL = 25
logging.addLevelName(THERMO_LEVEL, "THERMO")

def thermo_log(self, message, *args, **kwargs):
    if self.isEnabledFor(THERMO_LEVEL):
        self._log(THERMO_LEVEL, message, args, **kwargs)

logging.Logger.thermo = thermo_log

# Separate logger for thermo_log
thermo_file_handler = RotatingFileHandler(THERMO_FILE, maxBytes=10 * 1024 * 1024, backupCount=10)
thermo_format = logging.Formatter('%(message)s')
thermo_file_handler.setFormatter(thermo_format)  # Only log the message
thermo_file_handler.setLevel(THERMO_LEVEL)

thermo_logger = logging.getLogger('MD.THERMO')
thermo_logger.addHandler(console_handler)
thermo_logger.addHandler(thermo_file_handler)
thermo_logger.setLevel(THERMO_LEVEL)

thermostat_file_handler = RotatingFileHandler(THERMOSTAT_FILE, maxBytes=10 * 1024 * 1024, backupCount=10)
thermostat_file_handler.setFormatter(thermo_format)
thermostat_file_handler.setLevel(THERMO_LEVEL)

thermostat_logger = logging.getLogger('MD.THERMOSTAT')
thermostat_logger.addHandler(console_handler)
thermostat_logger.addHandler(thermostat_file_handler)
thermostat_logger.setLevel(THERMO_LEVEL)


## Compute some global constants and parameters
## ============================================

l_cell_x, l_cell_y, l_cell_z = r_cut/2, r_cut/2, r_cut/2

xlen = xhi - xlo
ylen = yhi - ylo
zlen = zhi - zlo
# \rho' = \lambda^D * \rho
# \rho = N_0/V_0
# N_0 = 1  For a simple cube
# V = 1 For a unit cell
# \lambda  Multiplicative factor
# D = 2, 3  Dimension of the simlation

lambda_ = np.power(1/rho, 1/3)
x_min = lambda_ * xlo
x_max = lambda_ * xhi
y_min = lambda_ * ylo
y_max = lambda_ * yhi
z_min = lambda_ * zlo
z_max = lambda_ * zhi

x_lenght = x_max - x_min
y_lenght = y_max - y_min
z_lenght = z_max - z_min

VOL = x_lenght * y_lenght * z_lenght

n_cell_x = int(x_lenght / l_cell_x)
n_cell_y = int(y_lenght / l_cell_y)
n_cell_z = int(z_lenght / l_cell_z)
n_cell = n_cell_x * n_cell_y * n_cell_z
cell_x = x_lenght / n_cell_x
cell_y = y_lenght / n_cell_y
cell_z = z_lenght / n_cell_z

n_cut_x = np.ceil(r_cut/cell_x)
n_cut_y = np.ceil(r_cut/cell_y)
n_cut_z = np.ceil(r_cut/cell_z)

r_cut2 = r_cut**2
v_cut = 4 * (np.power(r_cut, -12) - np.power(r_cut, -6))
f_cut = 24 * (2 * np.power(r_cut, -13) - np.power(r_cut, -7))
xi = 0
cell_list = {}
verlet_list = []


logger.info(f'number of cells: {n_cell_x} x {n_cell_y} x {n_cell_z}')
logger.info(f'Box dimensions: {x_lenght} x {y_lenght} x {z_lenght}')
logger.info(f'cutoff: {r_cut}')


## Funtions of the md
## ==================

def _init_positions(df: pd.DataFrame):
    x_coords = np.arange(x_min, x_max, lambda_)
    y_coords = np.arange(y_min, y_max, lambda_)
    z_coords = np.arange(z_min, z_max, lambda_)

    global N
    N = x_coords.size * y_coords.size * z_coords.size
    logger.info(f'Number of atoms: {N}')

    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)

    x_coords = np.reshape(x_coords, (N))
    y_coords = np.reshape(y_coords, (N))
    z_coords = np.reshape(z_coords, (N))

    df.x = x_coords + l_cell_x/2
    df.y = y_coords + l_cell_y/2
    df.z = z_coords + l_cell_z/2


def _init_velocities(df: pd.DataFrame):
    df[['vx', 'vy', 'vz']] = rng.normal(0, T0, (N, 3))
    df[['vx', 'vy', 'vz']] -= df[['vx', 'vy', 'vz']].mean()



def load_data() -> Tuple[pd.DataFrame, int]:
    df = pd.DataFrame(columns=[     'x',      'y',     'z',
                                   'vx',     'vy',    'vz',
                                   'ax',     'ay',    'az',
                               'cell_x', 'cell_y', 'cell_z'])
    k = -1
    fname = ''
    for d in OUT_PATH.iterdir():
        if (m := re.match(r'out_(\d+).csv', d.name)):
            n = int(m.group(1))
            if n > k:
                k = n
                fname = d

    if k < 0:
        _init_positions(df)
        _init_velocities(df)
        logger.info('First cells list update')
        update_cell(df)
        save_df(df, 0)

        logger.info('First acceleration update')
        V, VRL = update_acceleration3(df)
        thermo(df, V, VRL)

        return df, 0

    df[['x', 'y', 'z', 'vx', 'vy', 'vz',]] = pd.read_csv(fname, index_col=0)
    global N
    N = df.shape[0]
    
    update_cell(df)
    logger.info('First acceleration update')
    V, VRL = update_acceleration3(df)

    if THERMOSTAT_FILE.exists():
        global xi
        lines = open(THERMOSTAT_FILE).readlines()
        xi = float(lines[-1])

    return df, k


def init() -> pd.DataFrame:
    df = pd.DataFrame(columns=[     'x',      'y',     'z',
                                   'vx',     'vy',    'vz',
                                   'ax',     'ay',    'az',
                               'cell_x', 'cell_y', 'cell_z'])
    _init_positions(df)
    _init_velocities(df)

    return df


def update_cell(df: pd.DataFrame):
    logger.info('Updating cells list ...')

    df.cell_x = ((df.x - x_min) // cell_x).astype(int)
    df.cell_y = ((df.y - y_min) // cell_y).astype(int)
    df.cell_z = ((df.z - z_min) // cell_z).astype(int)

    logger.info('Cells list updated')


def update_list_generator1() -> Generator[int, int, int]:
    for x in range(n_cell_x):
        for y in range(n_cell_y):
            for z in range(n_cell_z):
                yield x, y, z


def compute_force_and_potential(dx: pd.Series, dy: pd.Series, dz: pd.Series,
                                r2: pd.Series) -> Tuple[pd.Series, pd.Series,
                                                        pd.Series, float, float]:
    '''Compute force, potential and virial

    $ V(r) = 4 * (1/r^12 - 1/r^6) - V(r_cutoff) - (r - r_cutoff) dV/dr|_r=r_cutoff $

    $ F(r) = -dV/dr = 24 * (2/r^13 - 1/r^7) - F(r_cutoff) $
    $ F_i(r) = F(r) * di/r = F(r) / r * di $

    
    '''
    ri12 = np.power(r2, -6)
    ri6 = np.power(r2, -3)
    r = np.sqrt(r2)

    v = 4 * (ri12 - ri6) - v_cut + (r - r_cut) * f_cut
    f = 24 * (2 * ri12 - ri6) / r2 - f_cut / r

    fx = f * dx
    fy = f * dy
    fz = f * dz

    vrl = fx*dx + fy*dy + fz*dz

    return fx, fy, fz, v.sum(), vrl.sum()


def get_mask_limits(serie: pd.Series, a: int, b: int, N: int):
    if a > 0 and b <= N:
        return (serie >= a) & (serie <= b)

    a = a % N
    b = b % N

    return (serie >= a) | (serie <= b)


def update_acceleration2(df: pd.DataFrame) -> Tuple[float, float]:
    logger.info('Updating accelerations ...')

    global verlet_list
    verlet_list = []

    df.ax = 0.0
    df.ay = 0.0
    df.az = 0.0
    V = 0.0
    VRL = 0.0

    df_copy = df.copy()
    for cx1, cy1, cz1 in tqdm(update_list_generator1(), desc=f'cells',
                              total=n_cell, leave=False, disable=True):
        logger.debug(f'cell 1: ({cx1}, {cy1}, {cz1})')
        
        index1 = df_copy[(df_copy.cell_x == cx1) & (df_copy.cell_y == cy1) & (df_copy.cell_z == cz1)].index

        if index1.empty:
            continue

        ax, bx = cx1 - n_cut_x, cx1 + n_cut_x
        ay, by = cy1 - n_cut_y, cy1 + n_cut_y
        az, bz = cz1 - n_cut_z, cz1 + n_cut_z

        mask_x = get_mask_limits(df_copy.cell_x, ax, bx, n_cell_x)
        mask_y = get_mask_limits(df_copy.cell_y, ay, by, n_cell_y)
        mask_z = get_mask_limits(df_copy.cell_z, az, bz, n_cell_z)

        df1 = df_copy.loc[mask_x & mask_y & mask_z, ['x', 'y', 'z', 'cell_x', 'cell_y', 'cell_z']].copy()

        df1.loc[df1.cell_x > bx, 'x'] -= x_lenght
        df1.loc[df1.cell_x < ax, 'x'] += x_lenght
        df1.loc[df1.cell_y > by, 'y'] -= y_lenght
        df1.loc[df1.cell_y < ay, 'y'] += y_lenght
        df1.loc[df1.cell_z > bz, 'z'] -= z_lenght
        df1.loc[df1.cell_z < az, 'z'] += z_lenght

        for i in index1:
            df1.drop(i, inplace=True)
            dx = df_copy.loc[i, 'x'] - df1.x
            dy = df_copy.loc[i, 'y'] - df1.y
            dz = df_copy.loc[i, 'z'] - df1.z

            r2 = dx*dx + dy*dy + dz*dz

            mask = r2 < r_cut2

            if not mask.any():
                continue

            verlet_list.append((i, mask))

            fx, fy, fz, v, vrl = compute_force_and_potential(dx[mask], dy[mask], dz[mask], r2[mask])
            df.loc[i, 'ax'] += fx.sum()
            df.loc[i, 'ay'] += fy.sum()
            df.loc[i, 'az'] += fz.sum()

            index2 = fx.index
            df.loc[index2, 'ax'] -= fx
            df.loc[index2, 'ay'] -= fy
            df.loc[index2, 'az'] -= fz

            V += v
            VRL += vrl
        
        df_copy.drop(index1, inplace=True)

    logger.info('Accelerations updated')
    return V, VRL


def update_acceleration3(df: pd.DataFrame) -> Tuple[float, float]:
    logger.info('Updating accelerations ...')

    global cell_list, verlet_list
    cell_list = []
    verlet_list = []

    df.ax = 0.0
    df.ay = 0.0
    df.az = 0.0
    V = 0.0
    VRL = 0.0

    df_copy = df.copy()
    for cx1, cy1, cz1 in tqdm(update_list_generator1(), desc=f'cells',
                              total=n_cell, leave=False, disable=True):
        logger.debug(f'cell 1: ({cx1}, {cy1}, {cz1})')
        mask1 = (df_copy.cell_x == cx1) & (df_copy.cell_y == cy1) & (df_copy.cell_z == cz1)
        index1 = df_copy[mask1].index

        if index1.empty:
            continue

        ax, bx = cx1 - n_cut_x, cx1 + n_cut_x
        ay, by = cy1 - n_cut_y, cy1 + n_cut_y
        az, bz = cz1 - n_cut_z, cz1 + n_cut_z

        mask_x = get_mask_limits(df_copy.cell_x, ax, bx, n_cell_x)
        mask_y = get_mask_limits(df_copy.cell_y, ay, by, n_cell_y)
        mask_z = get_mask_limits(df_copy.cell_z, az, bz, n_cell_z)
        mask_xyz = mask_x & mask_y & mask_z

        df1 = df_copy.loc[mask_xyz, ['x', 'y', 'z', 'cell_x', 'cell_y', 'cell_z']].copy()

        mask_xh = df1.cell_x > bx
        mask_xl = df1.cell_x < ax
        mask_yh = df1.cell_y > by
        mask_yl = df1.cell_y < ay
        mask_zh = df1.cell_z > bz
        mask_zl = df1.cell_z < az

        df1.loc[mask_xh, 'x'] -= x_lenght
        df1.loc[mask_xl, 'x'] += x_lenght
        df1.loc[mask_yh, 'y'] -= y_lenght
        df1.loc[mask_yl, 'y'] += y_lenght
        df1.loc[mask_zh, 'z'] -= z_lenght
        df1.loc[mask_zl, 'z'] += z_lenght
        masks_hl = (mask_xh, mask_xl, mask_yh, mask_yl, mask_zh, mask_zl)

        dx = df_copy.loc[index1, 'x'].to_numpy() - df1.x.to_numpy()[:, np.newaxis]
        dy = df_copy.loc[index1, 'y'].to_numpy() - df1.y.to_numpy()[:, np.newaxis]
        dz = df_copy.loc[index1, 'z'].to_numpy() - df1.z.to_numpy()[:, np.newaxis]

        r2 = dx*dx + dy*dy + dz*dz

        mask = (r2 < r_cut2)
        for i, ii in enumerate(index1, start=1):
            mask[df1.index == ii, :i] = False

        cell_list.append((cx1, cy1, cz1, index1, mask_xyz, mask, masks_hl))

        fx = np.zeros_like(r2)
        fy = np.zeros_like(r2)
        fz = np.zeros_like(r2)

        fx[mask], fy[mask], fz[mask], v, vrl = compute_force_and_potential(dx[mask], dy[mask], dz[mask], r2[mask])
        df.loc[index1, 'ax'] += fx.sum(axis=0)
        df.loc[index1, 'ay'] += fy.sum(axis=0)
        df.loc[index1, 'az'] += fz.sum(axis=0)

        df.loc[df1.index, 'ax'] -= fx.sum(axis=1)
        df.loc[df1.index, 'ay'] -= fy.sum(axis=1)
        df.loc[df1.index, 'az'] -= fz.sum(axis=1)

        V += v
        VRL += vrl

        df_copy.drop(index1, inplace=True)

    logger.info('Accelerations updated')
    return V, VRL


def update_acceleration3_verlet(df: pd.DataFrame) -> Tuple[float, float]:
    logger.info('Updating accelerations using verlet and cell list...')

    global cell_list, verlet_list

    df.ax = 0.0
    df.ay = 0.0
    df.az = 0.0
    V = 0.0
    VRL = 0.0

    df_copy = df.copy()
    for l in cell_list:
        cx1, cy1, cz1, index1, mask_xyz, mask, masks_hl = l
        logger.debug(f'cell 1: ({cx1}, {cy1}, {cz1})')

        ax, bx = cx1 - n_cut_x, cx1 + n_cut_x
        ay, by = cy1 - n_cut_y, cy1 + n_cut_y
        az, bz = cz1 - n_cut_z, cz1 + n_cut_z

        df1 = df_copy.loc[mask_xyz, ['x', 'y', 'z', 'cell_x', 'cell_y', 'cell_z']].copy()

        df1.loc[masks_hl[0], 'x'] -= x_lenght
        df1.loc[masks_hl[1], 'x'] += x_lenght
        df1.loc[masks_hl[2], 'y'] -= y_lenght
        df1.loc[masks_hl[3], 'y'] += y_lenght
        df1.loc[masks_hl[4], 'z'] -= z_lenght
        df1.loc[masks_hl[5], 'z'] += z_lenght

        dx = df_copy.loc[index1, 'x'].to_numpy() - df1.x.to_numpy()[:, np.newaxis]
        dy = df_copy.loc[index1, 'y'].to_numpy() - df1.y.to_numpy()[:, np.newaxis]
        dz = df_copy.loc[index1, 'z'].to_numpy() - df1.z.to_numpy()[:, np.newaxis]

        r2 = dx*dx + dy*dy + dz*dz

        # mask = (r2 < r_cut2)
        # for i, ii in enumerate(index1, start=1):
        #     mask[df1.index == ii, :i] = False

        fx = np.zeros_like(r2)
        fy = np.zeros_like(r2)
        fz = np.zeros_like(r2)

        fx[mask], fy[mask], fz[mask], v, vrl = compute_force_and_potential(dx[mask], dy[mask], dz[mask], r2[mask])
        df.loc[index1, 'ax'] += fx.sum(axis=0)
        df.loc[index1, 'ay'] += fy.sum(axis=0)
        df.loc[index1, 'az'] += fz.sum(axis=0)

        df.loc[df1.index, 'ax'] -= fx.sum(axis=1)
        df.loc[df1.index, 'ay'] -= fy.sum(axis=1)
        df.loc[df1.index, 'az'] -= fz.sum(axis=1)

        V += v
        VRL += vrl

        df_copy.drop(index1, inplace=True)

    logger.info('Accelerations updated')
    return V, VRL


def calc_velocity(v0: np.ndarray, a: np.ndarray, dt: float) -> np.ndarray:
    return v0 + a * dt


def update_velocity(df: pd.DataFrame, dt: float):
    logger.info('Updating velocities ...')

    df.vx = calc_velocity(df.vx, df.ax, dt)
    df.vy = calc_velocity(df.vy, df.ay, dt)
    df.vz = calc_velocity(df.vz, df.az, dt)

    logger.info('Velocities updated')


def calc_position(r0: np.ndarray, v0: np.ndarray, dt: float) -> np.ndarray:
    return r0 + v0 * dt


def update_position(df: pd.DataFrame, dt: float):
    logger.info('Updating positions ...')

    logger.info('\tCalculating new positions ...')
    df.x = calc_position(df.x, df.vx, dt)
    df.y = calc_position(df.y, df.vy, dt)
    df.z = calc_position(df.z, df.vz, dt)

    logger.info('\tCorrecting positions ...')
    mask_x = (df.x < x_min) | (df.x > x_max)
    mask_y = (df.y < y_min) | (df.y > y_max)
    mask_z = (df.z < z_min) | (df.z > z_max)
    df.loc[mask_x, 'x'] = (df.loc[mask_x, 'x'] - x_min) % x_lenght + x_min
    df.loc[mask_y, 'y'] = (df.loc[mask_y, 'y'] - y_min) % y_lenght + y_min
    df.loc[mask_z, 'z'] = (df.loc[mask_z, 'z'] - z_min) % z_lenght + z_min
    
    logger.info('Positions updated')


def save_df(df: pd.DataFrame, index: int):
    logger.info('saving data ...')

    df.drop(['ax', 'ay', 'az', 'cell_x', 'cell_y', 'cell_z'], axis=1).to_csv(OUT_PATH / f'out_{index}.csv')

    thermostat_logger.thermo(xi)
    
    logger.info('data saved')


def get_Ek_T(df: pd.DataFrame) -> Tuple[float, float]:
    v2 = np.power(df.vx, 2) + np.power(df.vy, 2) + np.power(df.vz, 2)
    v2 = v2.sum()
    Ek = v2 / 2  # m = 1
    T = v2 / 3 / N  # kb = 1
    
    return Ek, T


def get_pressure(T: float, vrl: float):
    '''Get the pressure using de Virial Theorem'''

    return (N * T + vrl / 3) / VOL


def thermo(df: pd.DataFrame, V: float, VRL: float):
    Ek, T = get_Ek_T(df)
    E = Ek + V
    P = get_pressure(T, VRL)

    thermo_logger.thermo(f'{E}\t{Ek}\t{V}\t{T}\t{P}')


def thermostat(df: pd.DataFrame, dt: float):
    v2 = np.square(df.vx) + np.square(df.vy) + np.square(df.vz)
    v2 = v2.sum()
    # x_i = dt * (2*E_k - 3*N*T_end)
    global xi
    xi += dt * (v2 / (3*N*T_end) - 1) / Q

    s = np.exp(- xi * dt)

    df.vx *= s
    df.vy *= s
    df.vz *= s


def main():
    logger.info('Initiating modelation')
    df, n0 = load_data()

    for i in tqdm(range(n0 + 1, n0 + STEPS + 1), desc='Steps'):
        logger.info(f'STEP: {i}')

        update_velocity(df, dt/2)
        thermostat(df, dt/2)

        update_position(df, dt)

        update_cell(df)
        V, VRL = update_acceleration3(df)
        # if i % 50 == 0:
        #     update_cell(df)
        #     V, VRL = update_acceleration3(df)
        # else:
        #     V, VRL = update_acceleration3_verlet(df)

        update_velocity(df, dt/2)
        thermostat(df, dt/2)

        if i % save_n == 0:
            save_df(df, i)

        if i % thermo_n == 0:
            thermo(df, V, VRL)


if __name__ == '__main__':
    main()

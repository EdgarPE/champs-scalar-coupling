import numpy as np
import pandas as pd
import os
import sys

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t5_lib import *

##### COPY__PASTE__LIB__END #####

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../feature/t5/crane'

SEED = 55
np.random.seed(SEED)


def t5_crane_features(structures):
    """
    https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    """

    # electronegativity and atomic_radius
    # from tqdm import tqdm_notebook as tqdm
    atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}  # Without fudge factor

    fudge_factor = 0.05
    atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
    # print(atomic_radius)

    electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

    # structures = pd.read_csv(structures, dtype={'atom_index':np.int8})

    atoms = structures['atom'].values
    atoms_en = [electronegativity[x] for x in atoms]
    atoms_rad = [atomic_radius[x] for x in atoms]

    structures['EN'] = atoms_en
    structures['rad'] = atoms_rad

    # print(structures.head())

    i_atom = structures['atom_index'].values
    p = structures[['x', 'y', 'z']].values
    p_compare = p
    m = structures['molecule_name'].values
    m_compare = m
    r = structures['rad'].values
    r_compare = r

    source_row = np.arange(len(structures))
    max_atoms = 28

    bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
    bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

    print('Calculating bonds')

    for i in range(max_atoms - 1):
        p_compare = np.roll(p_compare, -1, axis=0)
        m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)

        mask = np.where(m == m_compare, 1, 0)  # Are we still comparing atoms in the same molecule?
        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare

        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        target_row = source_row + i + 1  # Note: Will be out of bounds of bonds array for some values of i
        target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(structures),
                              target_row)  # If invalid target, write to dummy row

        source_atom = i_atom
        target_atom = i_atom + i + 1  # Note: Will be out of bounds of bonds array for some values of i
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0), max_atoms,
                               target_atom)  # If invalid target, write to dummy col

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

    print('Counting and condensing bonds')

    bonds_numeric = [[i for i, x in enumerate(row) if x] for row in bonds]
    bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]] for j, row in
                    enumerate(bond_dists)]
    bond_lengths_mean = [np.mean(x) for x in bond_lengths]
    bond_lengths_median = [np.median(x) for x in bond_lengths]
    bond_lengths_std = [np.std(x) for x in bond_lengths]
    n_bonds = [len(x) for x in bonds_numeric]

    # bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
    # bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

    bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean,
                 'bond_lengths_std': bond_lengths_std, 'bond_lengths_median': bond_lengths_median}
    bond_df = pd.DataFrame(bond_data)

    # print(bond_df.head(20))

    float32 = ['bond_lengths_mean', 'bond_lengths_std', 'bond_lengths_median']
    bond_df[float32] = bond_df[float32].astype('float32')

    int8 = ['n_bonds']
    bond_df[int8] = bond_df[int8].astype('int8')

    structures = structures.join(bond_df)

    new_columns = ['EN', 'rad', 'n_bonds', 'bond_lengths_median', 'bond_lengths_std', 'bond_lengths_mean']

    return structures[new_columns]


train, test, structures, contributions = t5_load_data(INPUT_DIR)

crane = t5_crane_features(structures)

# print(crane.shape)
# print(crane.dtypes.T)

crane.to_csv(f'{OUTPUT_DIR}/crane_structures.csv', index=False)
crane.to_parquet(f'{OUTPUT_DIR}/crane_structures.parquet', index=False)

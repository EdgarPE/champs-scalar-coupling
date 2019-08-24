import numpy as np
import pandas as pd
import os
import sys

pd.set_option('display.max_rows', 500)

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t5_lib import *

##### COPY__PASTE__LIB__END #####

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../feature/t5/giba'

SEED = 55
np.random.seed(SEED)

for dataset in ['train', 'test']:
    giba = pd.read_csv(INPUT_DIR + f'/giba/{dataset}_giba.csv')

    giba.drop(giba.columns[0], axis=1, inplace=True)

    drop = ['molecule_name', 'atom_index_1', 'atom_index_0', 'id', 'type', 'scalar_coupling_constant', 'ID',
            'structure_atom_0', 'structure_x_0', 'structure_y_0', 'structure_z_0', 'structure_atom_1', 'structure_x_1',
            'structure_y_1', 'structure_z_1', 'molecule_name.1', 'atom_index_1.1', 'pos']
    giba.drop(columns=drop, inplace=True)

    int16 = ['typei', 'N1', 'N2', 'link0', 'link1', 'linkN', ]
    giba[int16] = giba[int16].astype('int16')

    float32 = ['dist_xyz', 'inv_dist0', 'inv_dist1', 'inv_distP', 'R0', 'R1', 'E0', 'E1', 'inv_dist0R', 'inv_dist1R',
               'inv_distPR', 'inv_dist0E', 'inv_dist1E', 'inv_distPE', 'linkM0', 'linkM1',
               'min_molecule_atom_0_dist_xyz', 'mean_molecule_atom_0_dist_xyz', 'max_molecule_atom_0_dist_xyz',
               'sd_molecule_atom_0_dist_xyz', 'min_molecule_atom_1_dist_xyz', 'mean_molecule_atom_1_dist_xyz',
               'max_molecule_atom_1_dist_xyz', 'sd_molecule_atom_1_dist_xyz', 'coulomb_C.x', 'coulomb_F.x',
               'coulomb_H.x', 'coulomb_N.x', 'coulomb_O.x', 'yukawa_C.x', 'yukawa_F.x', 'yukawa_H.x', 'yukawa_N.x',
               'yukawa_O.x', 'vander_C.x', 'vander_F.x', 'vander_H.x', 'vander_N.x', 'vander_O.x', 'coulomb_C.y',
               'coulomb_F.y', 'coulomb_H.y', 'coulomb_N.y', 'coulomb_O.y', 'yukawa_C.y', 'yukawa_F.y', 'yukawa_H.y',
               'yukawa_N.y', 'yukawa_O.y', 'vander_C.y', 'vander_F.y', 'vander_H.y', 'vander_N.y', 'vander_O.y',
               'distC0', 'distH0', 'distN0', 'distC1', 'distH1', 'distN1', 'adH1', 'adH2', 'adH3', 'adH4', 'adC1',
               'adC2', 'adC3', 'adC4', 'adN1', 'adN2', 'adN3', 'adN4', ]
    giba[float32] = giba[float32].astype('float32')

    category = ['NC', 'NH', 'NN', 'NF', 'NO']
    giba[category] = giba[category].astype('category')

    # giba.rename(columns={c: f'giba_{c}' for c in giba.columns}, inplace=True)
    giba.columns = [f'giba_{c}' for c in giba.columns]

    # print(giba.dtypes.T)
    # print(giba.describe().T)
    # print(disp_mem_usage())
    # print(giba.describe().T)
    # print(giba.head(20))

    giba.to_csv(f'{OUTPUT_DIR}/{dataset}_giba.csv', index=False)
    giba.to_parquet(f'{OUTPUT_DIR}/{dataset}_giba.parquet', index=False)

    print(giba.shape)
    print(list(giba.columns))

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
OUTPUT_DIR = '../feature/t5/criskiev'

SEED = 55
np.random.seed(SEED)


def t5_criskiev_features(train_, test_, structures_):
    def add_coordinates(base, structures, index):
        df = pd.merge(base, structures, how='inner',
                      left_on=['molecule_index', f'atom_index_{index}'],
                      right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
        df = df.rename(columns={
            'atom': f'atom_{index}',
            'x': f'x_{index}',
            'y': f'y_{index}',
            'z': f'z_{index}'
        })
        return df

    def add_atoms(base, atoms):
        df = pd.merge(base, atoms, how='inner',
                      on=['molecule_index', 'atom_index_0', 'atom_index_1'])
        return df

    def merge_all_atoms(base, structures):
        df = pd.merge(base, structures, how='left',
                      left_on=['molecule_index'],
                      right_on=['molecule_index'])
        df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
        return df

    def add_center(df):
        df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
        df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
        df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

    def add_distance_to_center(df):
        df['d_c'] = ((
                             (df['x_c'] - df['x']) ** np.float32(2) +
                             (df['y_c'] - df['y']) ** np.float32(2) +
                             (df['z_c'] - df['z']) ** np.float32(2)
                     ) ** np.float32(0.5))

    def add_distance_between(df, suffix1, suffix2):
        df[f'd_{suffix1}_{suffix2}'] = ((
                                                (df[f'x_{suffix1}'] - df[f'x_{suffix2}']) ** np.float32(2) +
                                                (df[f'y_{suffix1}'] - df[f'y_{suffix2}']) ** np.float32(2) +
                                                (df[f'z_{suffix1}'] - df[f'z_{suffix2}']) ** np.float32(2)
                                        ) ** np.float32(0.5))

    def add_distances(df):
        n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])

        for i in range(1, n_atoms):
            for vi in range(min(4, i)):
                add_distance_between(df, i, vi)

    def build_type_dataframes(base, structures):
        base = base.drop('type', axis=1).copy()
        base = base.reset_index()
        base['id'] = base['id'].astype('int32')
        structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
        return base, structures

    def build_couple_dataframe(some_csv, structures_csv, n_atoms=10):
        base, structures = build_type_dataframes(some_csv, structures_csv)
        base = add_coordinates(base, structures, 0)
        base = add_coordinates(base, structures, 1)

        base = base.drop(['atom_0', 'atom_1'], axis=1)
        atoms = base.drop('id', axis=1).copy()
        if 'scalar_coupling_constant' in some_csv:
            atoms = atoms.drop(['scalar_coupling_constant'], axis=1)

        add_center(atoms)
        atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

        atoms = merge_all_atoms(atoms, structures)

        add_distance_to_center(atoms)

        atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
        atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
        atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
        atoms['num'] = atom_groups.cumcount() + 2
        atoms = atoms.drop(['d_c'], axis=1)
        atoms = atoms[atoms['num'] < n_atoms]

        atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
        atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
        atoms = atoms.reset_index()

        # downcast back to int8
        for col in atoms.columns:
            if col.startswith('atom_'):
                atoms[col] = atoms[col].fillna(0).astype('int8')

        atoms['molecule_index'] = atoms['molecule_index'].astype('int32')

        full = add_atoms(base, atoms)
        add_distances(full)

        full.sort_values('id', inplace=True)

        return full

    def take_n_atoms(df, n_atoms, four_start=4):
        labels = []
        for i in range(2, n_atoms):
            label = f'atom_{i}'
            labels.append(label)

        for i in range(n_atoms):
            num = min(i, 4) if i < four_start else 4
            for j in range(num):
                labels.append(f'd_{i}_{j}')
        if 'scalar_coupling_constant' in df:
            labels.append('scalar_coupling_constant')
        return df[labels]

    train = train_.copy(deep=True)
    test = test_.copy(deep=True)
    structures = structures_.copy(deep=True)

    train_dtypes = {
        'molecule_name': 'category',
        'atom_index_0': 'int8',
        'atom_index_1': 'int8',
        'type': 'category',
        'scalar_coupling_constant': 'float32'
    }

    for c in train_dtypes:
        train[c] = train[c].astype(train_dtypes[c])
        if c in test:
            test[c] = test[c].astype(train_dtypes[c])

    structures_dtypes = {
        'molecule_name': 'category',
        'atom_index': 'int8',
        'atom': 'category',
        'x': 'float32',
        'y': 'float32',
        'z': 'float32'
    }

    for c in structures_dtypes:
        structures[c] = structures[c].astype(structures_dtypes[c])

    ATOMIC_NUMBERS = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9
    }

    train.set_index('id', inplace=True)
    test.set_index('id', inplace=True)

    train['molecule_index'] = train.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    train = train[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

    test['molecule_index'] = test.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    test = test[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

    structures['molecule_index'] = structures.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    structures = structures[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
    structures['atom'] = structures['atom'].replace(ATOMIC_NUMBERS).astype('int8')

    train = build_couple_dataframe(train, structures)
    train = take_n_atoms(train, 10)
    train = train.fillna(0)
    train.index = train_.index

    test = build_couple_dataframe(test, structures)
    test = take_n_atoms(test, 10)
    test = test.fillna(0)
    test.index = test_.index

    criskiev_columns = ['atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'd_1_0',
                        'd_2_0', 'd_2_1', 'd_3_0', 'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0',
                        'd_5_1', 'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1', 'd_7_2',
                        'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1', 'd_9_2', 'd_9_3']

    # return pd.concat([train_, train[criskiev_columns]], axis=1), pd.concat([test_, test[criskiev_columns]], axis=1)
    return train[criskiev_columns], test[criskiev_columns]


train, test, structures, contributions = t5_load_data(INPUT_DIR)

train_, test_ = t5_criskiev_features(train, test, structures)

train_.to_csv(f'{OUTPUT_DIR}/criskiev_train.csv', index=False)
train_.to_parquet(f'{OUTPUT_DIR}/criskiev_train.parquet', index=False)
test_.to_csv(f'{OUTPUT_DIR}/criskiev_test.csv', index=False)
test_.to_parquet(f'{OUTPUT_DIR}/criskiev_test.parquet', index=False)

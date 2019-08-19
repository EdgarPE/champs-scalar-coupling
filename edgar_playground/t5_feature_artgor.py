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
OUTPUT_DIR = '../feature/t5/artgor'

SEED = 55
np.random.seed(SEED)


def t5_artgor_features(train, test):
    def artgor_features(df):
        """
        https://www.kaggle.com/artgor/using-meta-features-to-improve-model
        """

        old_columns = set(df.columns.copy())

        df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
        df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
        df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
        df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
        df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
        df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
        df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
        df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
        df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

        # Hiányzó, de nem lesz jobb
        # df[f'molecule_atom_index_0_x_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('mean')
        # df[f'molecule_atom_index_0_x_1_mean_diff'] = df[f'molecule_atom_index_0_x_1_mean'] - df['x_1']
        # df[f'molecule_atom_index_0_x_1_mean_div'] = df[f'molecule_atom_index_0_x_1_mean'] / df['x_1']
        # df[f'molecule_atom_index_0_x_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('max')
        # df[f'molecule_atom_index_0_x_1_max_diff'] = df[f'molecule_atom_index_0_x_1_max'] - df['x_1']

        df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
        df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
        df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
        df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
        df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

        # Hiányzó, de nem lesz jobb
        # df[f'molecule_atom_index_0_z_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('mean')
        # df[f'molecule_atom_index_0_z_1_mean_diff'] = df[f'molecule_atom_index_0_z_1_mean'] - df['z_1']
        # df[f'molecule_atom_index_0_z_1_mean_div'] = df[f'molecule_atom_index_0_z_1_mean'] / df['z_1']
        # df[f'molecule_atom_index_0_z_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('max')
        # df[f'molecule_atom_index_0_z_1_max_diff'] = df[f'molecule_atom_index_0_z_1_max'] - df['z_1']

        # Hiányzó, de nem lesz jobb
        # df[f'molecule_atom_index_1_x_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('mean')
        # df[f'molecule_atom_index_1_x_0_mean_diff'] = df[f'molecule_atom_index_1_x_0_mean'] - df['x_0']
        # df[f'molecule_atom_index_1_x_0_mean_div'] = df[f'molecule_atom_index_1_x_0_mean'] / df['x_0']
        # df[f'molecule_atom_index_1_x_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('max')
        # df[f'molecule_atom_index_1_x_0_max_diff'] = df[f'molecule_atom_index_1_x_0_max'] - df['x_0']
        #
        # Hiányzó, de nem lesz jobb
        # df[f'molecule_atom_index_1_y_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('mean')
        # df[f'molecule_atom_index_1_y_0_mean_diff'] = df[f'molecule_atom_index_1_y_0_mean'] - df['y_0']
        # df[f'molecule_atom_index_1_y_0_mean_div'] = df[f'molecule_atom_index_1_y_0_mean'] / df['y_0']
        # df[f'molecule_atom_index_1_y_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('max')
        # df[f'molecule_atom_index_1_y_0_max_diff'] = df[f'molecule_atom_index_1_y_0_max'] - df['y_0']
        #
        # Hiányzó, de nem lesz jobb
        # df[f'molecule_atom_index_1_z_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('mean')
        # df[f'molecule_atom_index_1_z_0_mean_diff'] = df[f'molecule_atom_index_1_z_0_mean'] - df['z_0']
        # df[f'molecule_atom_index_1_z_0_mean_div'] = df[f'molecule_atom_index_1_z_0_mean'] / df['z_0']
        # df[f'molecule_atom_index_1_z_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('max')
        # df[f'molecule_atom_index_1_z_0_max_diff'] = df[f'molecule_atom_index_1_z_0_max'] - df['z_0']

        df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
        df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
        df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

        df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
        df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
        df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

        df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
        df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
        df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

        df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
        df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
        df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

        df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
        df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
        df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

        df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
        df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
        df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

        df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
        df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
        df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

        df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
        df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
        df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

        df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

        df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
        df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
        df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

        df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
        df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

        df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
        df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

        df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
        df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
        df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

        df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
        df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
        df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
        df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

        int8 = ['molecule_couples', 'atom_0_couples_count', 'atom_1_couples_count']
        df[int8] = df[int8].astype('int8')

        new_columns = set(df.columns.copy()) - old_columns

        return df[new_columns]

    return artgor_features(train), artgor_features(test)


# Artgor needs dist, x_0, y_0, z_0, x_1, y_1, z_1
train, test, structures, contributions = t5_load_data(INPUT_DIR)
train, test = t5_merge_structures(train, test, structures)
t5_distance_feature(train, test)

train_, test_ = t5_artgor_features(train, test)

print(train_.shape)
print(train_.dtypes.T)

train_.to_csv(f'{OUTPUT_DIR}/artgor_train.csv', index=False)
train_.to_parquet(f'{OUTPUT_DIR}/artgor_train.parquet', index=False)
test_.to_csv(f'{OUTPUT_DIR}/artgor_test.csv', index=False)
test_.to_parquet(f'{OUTPUT_DIR}/artgor_test.parquet', index=False)

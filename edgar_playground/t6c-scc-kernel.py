#  --- HOME ---
INPUT_DIR = '../input/'
YUKAWA_DIR = '../input/yukawa'
FEATURE_DIR = '../feature/t6'
WORK_DIR = '../work/t6'
OUTPUT_DIR = '../work/t6'

#  --- KAGGLE ---
INPUT_DIR = '../input/champs-scalar-coupling'
YUKAWA_DIR = '../input/parallelization-of-coulomb-yukawa-interaction'
FEATURE_DIR = '../input/t6-features-parquet/t6_features_parquet'
WORK_DIR = '../input/t4abc-ua/t4_ua'
OUTPUT_DIR = '.'


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# exit(0)

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     print(dirname)
# exit(0)

import numpy as np
import pandas as pd
import os
import sys
import inspect
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
# pd.set_option('display.max_rows', 500)
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import gc
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization, Add, Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K


##### COPY__PASTE__LIB__BEGIN #####
def lineno():
    """Returns the current line number in our program."""
    return '[Line: %d]' % inspect.currentframe().f_back.f_lineno


def disp_mem_usage():
    # print(psutil.virtual_memory())  # physical memory usage
    print('[Memory, at line: %d] % 7.3f GB' % (
    inspect.currentframe().f_back.f_lineno, psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30))


def t6_load_data(input_dir):
    train = pd.read_csv(input_dir + '/train.csv')
    test = pd.read_csv(input_dir + '/test.csv')
    structures = pd.read_csv(input_dir + '/structures.csv')
    #     contributions = pd.read_csv(input_dir + '/scalar_coupling_contributions.csv')

    int32 = ['id']
    int8 = ['atom_index_0', 'atom_index_1']
    float32 = ['scalar_coupling_constant']

    train[int32] = train[int32].astype('int32')
    train[int8] = train[int8].astype('int8')

    train[float32] = train[float32].astype('float32')
    test[int32] = test[int32].astype('int32')
    test[int8] = test[int8].astype('int8')

    int8 = ['atom_index']
    float32 = ['x', 'y', 'z']

    structures[int8] = structures[int8].astype('int8')
    structures[float32] = structures[float32].astype('float32')

    #     int8 = ['atom_index_0', 'atom_index_1']
    #     float32 = ['fc', 'sd', 'pso', 'dso']

    #     contributions[int8] = contributions[int8].astype('int8')
    #     contributions[float32] = contributions[float32].astype('float32')

    return train, test, structures  # , contributions


def t6_load_submissions(input_dir):
    submissions = pd.read_csv(input_dir + '/sample_submission.csv')

    int32 = ['id']
    float32 = ['scalar_coupling_constant']

    submissions[int32] = submissions[int32].astype('int32')
    submissions[float32] = submissions[float32].astype('float32')

    return submissions


def t6_load_feature_criskiev(feature_dir, train_, test_, train_selector, test_selector):
    train = pd.read_parquet(feature_dir + '/criskiev/criskiev_train.parquet')[train_selector]
    test = pd.read_parquet(feature_dir + '/criskiev/criskiev_test.parquet')[test_selector]

    int8 = ['cris_atom_2', 'cris_atom_3', 'cris_atom_4', 'cris_atom_5', 'cris_atom_6', 'cris_atom_7', 'cris_atom_8',
            'cris_atom_9']

    train[int8] = train[int8].astype('int8')
    test[int8] = test[int8].astype('int8')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t6_load_feature_crane(feature_dir, structures):
    crane = pd.read_parquet(feature_dir + '/crane/crane_structures.parquet')

    return structures.join(crane)


def t6_load_feature_artgor(feature_dir, train_, test_, train_selector, test_selector):
    train = pd.read_parquet(feature_dir + '/artgor/artgor_train.parquet')[train_selector]
    test = pd.read_parquet(feature_dir + '/artgor/artgor_test.parquet')[test_selector]

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t6_load_feature_giba(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/giba/train_giba.parquet')
    test = pd.read_parquet(feature_dir + '/giba/test_giba.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


# https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
def t6_merge_yukawa(input_dir, structures):
    yukawa = pd.read_csv(input_dir + '/structures_yukawa.csv').astype('float32')
    yukawa.columns = [f'yuka_{c}' for c in yukawa.columns]
    structures = pd.concat([structures, yukawa], axis=1)

    return structures


def t6_load_feature_edgar(feature_dir, train_, test_, train_selector, test_selector):
    train = pd.read_parquet(feature_dir + '/edgar/edgar_train.parquet')[train_selector].astype('int8')
    test = pd.read_parquet(feature_dir + '/edgar/edgar_test.parquet')[test_selector].astype('int8')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t6_load_data_mulliken(input_dir):
    # dipole_moments = pd.read_csv(input_dir + '/dipole_moments.csv')
    # magnetic_st = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')
    mulliken_charges = pd.read_csv(input_dir + '/mulliken_charges.csv')
    # potential_energy = pd.read_csv(input_dir + '/potential_energy.csv')

    int8 = ['atom_index']
    float32 = ['mulliken_charge']

    mulliken_charges[int8] = mulliken_charges[int8].astype('int8')
    mulliken_charges[float32] = mulliken_charges[float32].astype('float32')

    return mulliken_charges


def t6_load_data_magnetic_st(input_dir):
    magnetic_st = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')

    drop = ['XY', 'XZ', 'YX', 'YZ', 'ZX', 'ZY']
    magnetic_st.drop(columns=drop, inplace=True)

    int8 = ['atom_index']
    magnetic_st[int8] = magnetic_st[int8].astype('int8')

    float32 = ['XX', 'YY', 'ZZ']
    magnetic_st[float32] = magnetic_st[float32].astype('float32')

    return magnetic_st


def t6_load_data_mulliken_oof(work_dir, train, test, train_selector, test_selector):
    mulliken_charges = pd.read_csv(work_dir + '/t4a_mulliken_train.csv')[train_selector]

    float32 = ['oof_mulliken_charge']
    mulliken_charges[float32] = mulliken_charges[float32].astype('float32')

    train = pd.merge(train, mulliken_charges, how='left',
                     left_on=['molecule_name', 'atom_index_0'],
                     right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)
    train.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_0'})

    train = pd.merge(train, mulliken_charges, how='left',
                     left_on=['molecule_name', 'atom_index_1'],
                     right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)
    train.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_1'})

    mulliken_charges = pd.read_csv(work_dir + '/t4a_mulliken_test.csv')[test_selector]

    float32 = ['oof_mulliken_charge']
    mulliken_charges[float32] = mulliken_charges[float32].astype('float32')

    test = pd.merge(test, mulliken_charges, how='left',
                    left_on=['molecule_name', 'atom_index_0'],
                    right_on=['molecule_name', 'atom_index'])
    test.drop('atom_index', axis=1, inplace=True)
    test.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_0'})

    test = pd.merge(test, mulliken_charges, how='left',
                    left_on=['molecule_name', 'atom_index_1'],
                    right_on=['molecule_name', 'atom_index'])
    test.drop('atom_index', axis=1, inplace=True)
    test.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_1'})

    return train, test


def t6_load_data_magnetic_st_oof(work_dir, train, test):
    # todo
    pass


def t6_merge_contributions(train, contributions):
    train = pd.merge(train, contributions, how='left',
                     left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                     right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

    # train['contrib_sum'] = train['fc'] + train['sd'] + train['pso'] + train['dso']

    return train


def t6_load_data_contributions_oof(work_dir, train, test, train_selector, test_selector):
    oof_contributions = pd.read_csv(work_dir + '/t4b_contributions_train.csv')[train_selector]

    float32 = ['oof_fc', 'oof_sd', 'oof_pso', 'oof_dso']
    oof_contributions[float32] = oof_contributions[float32].astype('float32')

    train = pd.merge(train, oof_contributions, how='left',
                     left_on=['id'],
                     right_on=['id'])
    train.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })
    train['contrib_sum'] = train['fc'] + train['sd'] + train['pso'] + train['dso']

    oof_contributions = pd.read_csv(work_dir + '/t4b_contributions_test.csv')[test_selector]

    float32 = ['oof_fc', 'oof_sd', 'oof_pso', 'oof_dso']
    oof_contributions[float32] = oof_contributions[float32].astype('float32')

    test = pd.merge(test, oof_contributions, how='left',
                    left_on=['id'],
                    right_on=['id'])
    test.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })
    test['contrib_sum'] = test['fc'] + test['sd'] + test['pso'] + test['dso']

    return train, test


def map_atom_info(df, atom_idx, structures):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'], suffixes=["_a0", "_a1"])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def t6_merge_structures(train_, test_, structures):
    train = map_atom_info(train_, 0, structures)
    train = map_atom_info(train, 1, structures)
    train.index = train_.index

    test = map_atom_info(test_, 0, structures)
    test = map_atom_info(test, 1, structures)
    test.index = test_.index

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist'] = 1 / (train['dist'] ** 3)  # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    test['dist'] = 1 / (test['dist'] ** 3)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    return train, test


def t6_distance_feature(train, test):
    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist_lin'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist_lin'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist'] = 1 / (train['dist_lin'] ** 3)  # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    test['dist'] = 1 / (test['dist_lin'] ** 3)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    # train['type_0'] = train['type'].apply(lambda x: x[0])
    # test['type_0'] = test['type'].apply(lambda x: x[0])

    train['subtype'] = (train['type'].eq('1JHC') & train['dist_lin'].gt(1.065)).astype('int8')
    test['subtype'] = (test['type'].eq('1JHC') & test['dist_lin'] > 1.065).astype('int8')


def t6_prepare_columns(train, test, good_columns_extra=None):
    good_columns = [
        # 'bond_lengths_mean_y',
        # 'bond_lengths_median_y',
        # 'bond_lengths_std_y',
        # 'bond_lengths_mean_x',

        'artg_molecule_atom_index_0_dist_min',
        'artg_molecule_atom_index_0_dist_max',
        'artg_molecule_atom_index_1_dist_min',
        'artg_molecule_atom_index_0_dist_mean',
        'artg_molecule_atom_index_0_dist_std',

        'artg_molecule_atom_index_1_dist_std',
        'artg_molecule_atom_index_1_dist_max',
        'artg_molecule_atom_index_1_dist_mean',
        'artg_molecule_atom_index_0_dist_max_diff',
        'artg_molecule_atom_index_0_dist_max_div',
        'artg_molecule_atom_index_0_dist_std_diff',
        'artg_molecule_atom_index_0_dist_std_div',
        'artg_atom_0_couples_count',
        'artg_molecule_atom_index_0_dist_min_div',
        'artg_molecule_atom_index_1_dist_std_diff',
        'artg_molecule_atom_index_0_dist_mean_div',
        'artg_atom_1_couples_count',
        'artg_molecule_atom_index_0_dist_mean_diff',
        'artg_molecule_couples',

        'artg_molecule_dist_mean',
        'artg_molecule_atom_index_1_dist_max_diff',
        'artg_molecule_atom_index_0_y_1_std',
        'artg_molecule_atom_index_1_dist_mean_diff',
        'artg_molecule_atom_index_1_dist_std_div',
        'artg_molecule_atom_index_1_dist_mean_div',
        'artg_molecule_atom_index_1_dist_min_diff',
        'artg_molecule_atom_index_1_dist_min_div',
        'artg_molecule_atom_index_1_dist_max_div',
        'artg_molecule_atom_index_0_z_1_std',

        'artg_molecule_type_dist_std_diff',
        'artg_molecule_atom_1_dist_min_diff',
        'artg_molecule_atom_index_0_x_1_std',
        'artg_molecule_dist_min',
        'artg_molecule_atom_index_0_dist_min_diff',
        'artg_molecule_atom_index_0_y_1_mean_diff',
        'artg_molecule_type_dist_min',
        'artg_molecule_atom_1_dist_min_div',

        'artg_molecule_dist_max',
        'artg_molecule_atom_1_dist_std_diff',
        'artg_molecule_type_dist_max',
        'artg_molecule_atom_index_0_y_1_max_diff',
        'artg_molecule_type_0_dist_std_diff',
        'artg_molecule_type_dist_mean_diff',
        'artg_molecule_atom_1_dist_mean',
        'artg_molecule_atom_index_0_y_1_mean_div',
        'artg_molecule_type_dist_mean_div',

        # 'atom_index_0',
        # 'atom_index_1',
        # 'type',
        # 'y_0',
        'dist',
        'dist_lin',
        # 'subtype',

        # Yukawa
        'yuka_dist_C_0_a0', 'yuka_dist_C_1_a0', 'yuka_dist_C_2_a0', 'yuka_dist_C_3_a0', 'yuka_dist_C_4_a0',
        'yuka_dist_F_0_a0', 'yuka_dist_F_1_a0', 'yuka_dist_F_2_a0', 'yuka_dist_F_3_a0', 'yuka_dist_F_4_a0',
        'yuka_dist_H_0_a0', 'yuka_dist_H_1_a0', 'yuka_dist_H_2_a0', 'yuka_dist_H_3_a0', 'yuka_dist_H_4_a0',
        'yuka_dist_N_0_a0', 'yuka_dist_N_1_a0', 'yuka_dist_N_2_a0', 'yuka_dist_N_3_a0', 'yuka_dist_N_4_a0',
        'yuka_dist_O_0_a0', 'yuka_dist_O_1_a0', 'yuka_dist_O_2_a0', 'yuka_dist_O_3_a0', 'yuka_dist_O_4_a0',

        'yuka_dist_C_0_a1', 'yuka_dist_C_1_a1', 'yuka_dist_C_2_a1', 'yuka_dist_C_3_a1', 'yuka_dist_C_4_a1',
        'yuka_dist_F_0_a1', 'yuka_dist_F_1_a1', 'yuka_dist_F_2_a1', 'yuka_dist_F_3_a1', 'yuka_dist_F_4_a1',
        'yuka_dist_H_0_a1', 'yuka_dist_H_1_a1', 'yuka_dist_H_2_a1', 'yuka_dist_H_3_a1', 'yuka_dist_H_4_a1',
        'yuka_dist_N_0_a1', 'yuka_dist_N_1_a1', 'yuka_dist_N_2_a1', 'yuka_dist_N_3_a1', 'yuka_dist_N_4_a1',
        'yuka_dist_O_0_a1', 'yuka_dist_O_1_a1', 'yuka_dist_O_2_a1', 'yuka_dist_O_3_a1', 'yuka_dist_O_4_a1',

        # Crane
        # 'cran_EN_a0', 'cran_rad_a0', 'cran_n_bonds_a0', 'cran_bond_lengths_mean_a0', 'cran_bond_lengths_std_a0',
        # 'cran_bond_lengths_median_a0',
        # 'cran_EN_a1', 'cran_rad_a1', 'cran_n_bonds_a1', 'cran_bond_lengths_mean_a1', 'cran_bond_lengths_std_a1',
        # 'cran_bond_lengths_median_a1',

        # Criskiev
        # 'd_1_0'
        'cris_atom_2', 'cris_atom_3', 'cris_atom_4', 'cris_atom_5', 'cris_atom_6', 'cris_atom_7', 'cris_atom_8',
        'cris_atom_9', 'cris_d_2_0', 'cris_d_2_1', 'cris_d_3_0', 'cris_d_3_1', 'cris_d_3_2', 'cris_d_4_0', 'cris_d_4_1',
        'cris_d_4_2', 'cris_d_4_3', 'cris_d_5_0', 'cris_d_5_1', 'cris_d_5_2', 'cris_d_5_3', 'cris_d_6_0', 'cris_d_6_1',
        'cris_d_6_2', 'cris_d_6_3', 'cris_d_7_0', 'cris_d_7_1', 'cris_d_7_2', 'cris_d_7_3', 'cris_d_8_0', 'cris_d_8_1',
        'cris_d_8_2', 'cris_d_8_3', 'cris_d_9_0', 'cris_d_9_1', 'cris_d_9_2', 'cris_d_9_3',
        'cris_d_2_min', 'cris_d_3_min', 'cris_d_4_min', 'cris_d_5_min', 'cris_d_6_min', 'cris_d_7_min', 'cris_d_8_min',
        'cris_d_9_min',

        # Criskiev extra
        # 'd_1_0_log', 'd_2_0_log', 'd_2_1_log', 'd_3_0_log', 'd_3_1_log', 'd_3_2_log', 'd_4_0_log', 'd_4_1_log',
        # 'd_4_2_log', 'd_4_3_log', 'd_5_0_log', 'd_5_1_log', 'd_5_2_log', 'd_5_3_log', 'd_6_0_log', 'd_6_1_log',
        # 'd_6_2_log', 'd_6_3_log', 'd_7_0_log', 'd_7_1_log', 'd_7_2_log', 'd_7_3_log', 'd_8_0_log', 'd_8_1_log',
        # 'd_8_2_log', 'd_8_3_log', 'd_9_0_log', 'd_9_1_log', 'd_9_2_log', 'd_9_3',
        #
        # 'd_1_0_recp', 'd_2_0_recp', 'd_2_1_recp', 'd_3_0_recp', 'd_3_1_recp', 'd_3_2_recp', 'd_4_0_recp', 'd_4_1_recp',
        # 'd_4_2_recp', 'd_4_3_recp', 'd_5_0_recp', 'd_5_1_recp', 'd_5_2_recp', 'd_5_3_recp', 'd_6_0_recp', 'd_6_1_recp',
        # 'd_6_2_recp', 'd_6_3_recp', 'd_7_0_recp', 'd_7_1_recp', 'd_7_2_recp', 'd_7_3_recp', 'd_8_0_recp', 'd_8_1_recp',
        # 'd_8_2_recp', 'd_8_3_recp', 'd_9_0_recp', 'd_9_1_recp', 'd_9_2_recp', 'd_9_3'

        # Giba
        # 'giba_typei', 'giba_N1', 'giba_N2', 'giba_link0', 'giba_link1', 'giba_linkN', 'giba_dist_xyz', 'giba_inv_dist0',
        # 'giba_inv_dist1', 'giba_inv_distP', 'giba_R0', 'giba_R1', 'giba_E0', 'giba_E1', 'giba_inv_dist0R',
        # 'giba_inv_dist1R', 'giba_inv_distPR', 'giba_inv_dist0E', 'giba_inv_dist1E', 'giba_inv_distPE', 'giba_linkM0',
        # 'giba_linkM1', 'giba_min_molecule_atom_0_dist_xyz', 'giba_mean_molecule_atom_0_dist_xyz',
        # 'giba_max_molecule_atom_0_dist_xyz', 'giba_sd_molecule_atom_0_dist_xyz', 'giba_min_molecule_atom_1_dist_xyz',
        # 'giba_mean_molecule_atom_1_dist_xyz', 'giba_max_molecule_atom_1_dist_xyz', 'giba_sd_molecule_atom_1_dist_xyz',
        # 'giba_coulomb_C.x', 'giba_coulomb_F.x', 'giba_coulomb_H.x', 'giba_coulomb_N.x', 'giba_coulomb_O.x',
        # 'giba_yukawa_C.x', 'giba_yukawa_F.x', 'giba_yukawa_H.x', 'giba_yukawa_N.x', 'giba_yukawa_O.x',
        # 'giba_vander_C.x', 'giba_vander_F.x', 'giba_vander_H.x', 'giba_vander_N.x', 'giba_vander_O.x',
        # 'giba_coulomb_C.y', 'giba_coulomb_F.y', 'giba_coulomb_H.y', 'giba_coulomb_N.y', 'giba_coulomb_O.y',
        # 'giba_yukawa_C.y', 'giba_yukawa_F.y', 'giba_yukawa_H.y', 'giba_yukawa_N.y', 'giba_yukawa_O.y',
        # 'giba_vander_C.y', 'giba_vander_F.y', 'giba_vander_H.y', 'giba_vander_N.y', 'giba_vander_O.y', 'giba_distC0',
        # 'giba_distH0', 'giba_distN0', 'giba_distC1', 'giba_distH1', 'giba_distN1', 'giba_adH1', 'giba_adH2',
        # 'giba_adH3', 'giba_adH4', 'giba_adC1', 'giba_adC2', 'giba_adC3', 'giba_adC4', 'giba_adN1', 'giba_adN2',
        # 'giba_adN3', 'giba_adN4', 'giba_NC', 'giba_NH', 'giba_NN', 'giba_NF', 'giba_NO',
    ]

    good_columns += (good_columns_extra if good_columns_extra is not None else [])

    X = train[good_columns].copy()
    X_test = test[good_columns].copy()

    return X, X_test  # labels


def t6_to_parquet(work_dir, train, test, structures, contributions):
    train.to_parquet(f'{work_dir}/t6_train.parquet')
    test.to_parquet(f'{work_dir}/t6_test.parquet')
    structures.to_parquet(f'{work_dir}/t6_structures.parquet')
    contributions.to_parquet(f'{work_dir}/t6_contributions.parquet')


def t6_read_parquet(work_dir):
    train = pd.read_parquet(f'{work_dir}/t6_train.parquet')
    test = pd.read_parquet(f'{work_dir}/t6_test.parquet')
    structures = pd.read_parquet(f'{work_dir}/t6_structures.parquet')
    contributions = pd.read_parquet(f'{work_dir}/t6_contributions.parquet')

    return train, test, structures, contributions


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
    #                 if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
    #                     df[col] = df[col].astype(np.float32)
    #                 else:
    #                     df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def create_nn_model(input_shape):
    inp = Input(shape=(input_shape,))
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation="linear")(x)
    # out1 = Dense(2, activation="linear")(x)#mulliken charge 2
    # out2 = Dense(6, activation="linear")(x)#tensor 6(xx,yy,zz)
    # out3 = Dense(12, activation="linear")(x)#tensor 12(others)
    # out4 = Dense(1, activation="linear")(x)#scalar_coupling_constant
    # model = Model(inputs=inp, outputs=[out,out1,out2,out3,out4])
    model = Model(inputs=inp, outputs=[out])
    return model


def plot_history(history, label):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _ = plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


##### COPY__PASTE__LIB__END #####

# TYPE_WL = ['1JHN', '2JHN', '3JHN', '2JHH', '3JHH', '1JHC', '2JHC', '3JHC', ]
TYPE_WL = ['3JHC', '2JHC', '1JHC', '3JHH', '2JHH', '3JHN', '2JHN', '1JHN' ]

# TARGET_WL = ['fc', 'sd', 'pso', 'dso']
TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

cv_score = []
cv_score_total = 0
epoch_n = 500
verbose = 1
batch_size = 2048

# Set up GPU preferences
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 2})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
K.set_session(sess)

start_time = datetime.now()

submission = t6_load_submissions(INPUT_DIR)

# Loop through each molecule type
for type_name in TYPE_WL:
    gc.collect()

    train, test, structures = t6_load_data(INPUT_DIR)
    print(lineno(), train.shape)

    train_selector = train['type'] == type_name
    test_selector = test['type'] == type_name
    train = train[train_selector]
    test = test[test_selector]

    target_data = train.loc[train_selector, 'scalar_coupling_constant'].values

    structures = t6_merge_yukawa(YUKAWA_DIR, structures)

    # structures = t6_load_feature_crane(FEATURE_DIR, structures)

    train, test = t6_merge_structures(train, test, structures)

    disp_mem_usage()
    del structures
    reduce_mem_usage(train)
    reduce_mem_usage(test)
    gc.collect()
    disp_mem_usage()

    train, test = t6_load_feature_criskiev(FEATURE_DIR, train, test, train_selector, test_selector)
    disp_mem_usage()

    t6_distance_feature(train, test)
    disp_mem_usage()

    train, test = t6_load_feature_artgor(FEATURE_DIR, train, test, train_selector, test_selector)
    disp_mem_usage()

    # train, test = t6_load_feature_edgar(FEATURE_DIR, train, test, train_selector, test_selector)
    # disp_mem_usage()
    # print(train.shape, test.shape)

    #
    # Load Phase 1. OOF data Mulliken charge
    #
    train, test = t6_load_data_mulliken_oof(WORK_DIR, train, test, train_selector, test_selector)
    disp_mem_usage()

    #
    # Load Phase 2. OOF data Contributions (fc, sd, pso, dso)
    #
    train, test = t6_load_data_contributions_oof(WORK_DIR, train, test, train_selector, test_selector)
    disp_mem_usage()

    # to_drop = ['type']
    # train.drop(columns=to_drop, inplace=True)
    # test.drop(columns=to_drop, inplace=True)

    extra_cols = []
    extra_cols += ['mulliken_charge_0', 'mulliken_charge_1']
    extra_cols += ['fc', 'sd', 'pso', 'dso', 'contrib_sum']
    # extra_cols += ['qcut_subtype_0', 'qcut_subtype_1', 'qcut_subtype_2']
    X_learn, X_predict = t6_prepare_columns(train, test, good_columns_extra=extra_cols)
    del train, test
    gc.collect()
    disp_mem_usage()

    test_prediction = np.zeros(len(X_predict))

    print('Training %s' % type_name, 'out of', TYPE_WL, '\n')

    X_learn = X_learn.fillna(0)
    X_predict = X_predict.fillna(0)
    input_features = list(X_learn.columns)

    # pd.set_option('display.max_rows', 500)
    # print(df_train_.dtypes.T)

    # print(lineno(), type_name)
    # is_nan = df_train_.isin([np.nan]).any(0)
    # print(list(df_train_.loc[:,is_nan].columns))
    # is_minf = df_train_.isin([-np.inf]).any(0)
    # print(list(df_train_.loc[:,is_minf].columns))
    # is_pinf = df_train_.isin([np.inf]).any(0)
    # print(list(df_train_.loc[:,is_pinf].columns))

    for df in [X_learn, X_predict]:
        df[df.isin([np.nan, -np.inf, np.inf])] = 0

    # Standard Scaler from sklearn does seem to work better here than other Scalers
    input_data = StandardScaler().fit_transform(
        pd.concat([X_learn.loc[:, input_features], X_predict.loc[:, input_features]]))
    # input_data=StandardScaler().fit_transform(df_train_.loc[:,input_features])

    # Simple split to provide us a validation set to do our CV checks with
    train_index, cv_index = train_test_split(np.arange(len(X_learn)), random_state=SEED, test_size=0.1)
    # Split all our input and targets by train and cv indexes
    train_target = target_data[train_index]
    cv_target = target_data[cv_index]
    train_input = input_data[train_index]
    cv_input = input_data[cv_index]
    test_input = input_data[len(X_learn):, :]

    del X_learn, X_predict
    gc.collect()

    # Build the Neural Net
    nn_model = create_nn_model(train_input.shape[1])

    nn_model.compile(loss='mae', optimizer=Adam())  # , metrics=[auc])

    gc.collect()
    disp_mem_usage()

    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=1, mode='auto',
                                 restore_best_weights=True)
    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-6, mode='auto', verbose=1)
    # Save the best value of the model for future use
    history = nn_model.fit(train_input, [train_target],
                           validation_data=(cv_input, [cv_target]),
                           callbacks=[es, rlr], epochs=epoch_n, batch_size=batch_size, verbose=verbose)

    cv_predict = nn_model.predict(cv_input)

    # plot_history(history, mol_type)
    accuracy = np.mean(np.abs(cv_target - cv_predict[:, 0]))
    print('CV for %s: %.4f' % (type_name, np.log(accuracy)))
    cv_score.append(np.log(accuracy))
    cv_score_total += np.log(accuracy)

    # Predict on the test data set using our trained model
    test_predict = nn_model.predict(test_input)

    del input_data, target_data
    del train_target, cv_target, train_input, cv_input
    del nn_model
    gc.collect()

    # for each molecule type we'll grab the predicted values
    # test_prediction[test["type"] == t] = test_predict[:, 0]
    # print(lineno(), submission.shape)
    # print(lineno(), test_selector.shape)
    # print(lineno(), submission[test_selector].shape)
    # print(lineno(), test_predict.shape)
    # print(lineno(), test_prediction.shape)
    submission.loc[test_selector, 'scalar_coupling_constant'] = test_predict
    K.clear_session()

cv_score_total /= len(TYPE_WL)

print('Total training time: ', datetime.now() - start_time)

i = 0
for type_name in TYPE_WL:
    print(type_name, ": cv score is ", cv_score[i])
    i += 1
print("total cv score is", cv_score_total)

# print(type(test_prediction))
print(test_prediction.shape)

submission.to_csv(f'{OUTPUT_DIR}/t6c_scc.csv', index=False)

# Add more layers to get a better score! However,maybe,features are really more important than algorithms...

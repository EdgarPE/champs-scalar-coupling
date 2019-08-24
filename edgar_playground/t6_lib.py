import numpy as np
import pandas as pd
import psutil
##### COPY__PASTE__LIB__BEGIN #####

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



def disp_mem_usage():
    # print(psutil.virtual_memory())  # physical memory usage
    print('[Memory] % 7.3f GB' % (psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30))


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


def t6_load_data(input_dir):
    train = pd.read_csv(input_dir + '/train.csv')
    test = pd.read_csv(input_dir + '/test.csv')
    structures = pd.read_csv(input_dir + '/structures.csv')
    contributions = pd.read_csv(input_dir + '/scalar_coupling_contributions.csv')

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

    int8 = ['atom_index_0', 'atom_index_1']
    float32 = ['fc', 'sd', 'pso', 'dso']

    contributions[int8] = contributions[int8].astype('int8')
    contributions[float32] = contributions[float32].astype('float32')

    return train, test, structures, contributions


def t6_load_submissions(input_dir):
    submissions = pd.read_csv(input_dir + '/sample_submission.csv')

    int32 = ['id']
    float32 = ['scalar_coupling_constant']

    submissions[int32] = submissions[int32].astype('int32')
    submissions[float32] = submissions[float32].astype('float32')

    return submissions


def t6_load_feature_criskiev(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/criskiev/criskiev_train.parquet')
    test = pd.read_parquet(feature_dir + '/criskiev/criskiev_test.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t6_load_feature_crane(feature_dir, structures):
    crane = pd.read_parquet(feature_dir + '/crane/crane_structures.parquet')

    return structures.join(crane)


def t6_load_feature_artgor(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/artgor/artgor_train.parquet')
    test = pd.read_parquet(feature_dir + '/artgor/artgor_test.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t6_load_feature_giba(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/giba/train_giba.parquet')
    test = pd.read_parquet(feature_dir + '/giba/test_giba.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


# https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
def t6_merge_yukawa(input_dir, structures):
    yukawa = pd.read_csv(input_dir + '/yukawa/structures_yukawa.csv').astype('float32')
    yukawa.columns = [f'yuka_{c}' for c in yukawa.columns]
    structures = pd.concat([structures, yukawa], axis=1)

    return structures


def t6_load_feature_edgar(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/edgar/edgar_train.parquet')
    test = pd.read_parquet(feature_dir + '/edgar/edgar_test.parquet')

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


def t6_load_data_mulliken_oof(work_dir, train, test):
    mulliken_charges = pd.read_csv(work_dir + '/t6a_mulliken_train.csv')
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

    mulliken_charges = pd.read_csv(work_dir + '/t6a_mulliken_test.csv')
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


def t6_load_data_contributions_oof(work_dir, train, test):
    oof_contributions = pd.read_csv(work_dir + '/t6b_contributions_train.csv')
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

    oof_contributions = pd.read_csv(work_dir + '/t6b_contributions_test.csv')
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


def t6_merge_structures(train, test, structures):
    train = map_atom_info(train, 0, structures)
    train = map_atom_info(train, 1, structures)

    test = map_atom_info(test, 0, structures)
    test = map_atom_info(test, 1, structures)

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

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

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

        'dist',
        'dist_lin',
        'subtype',

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

        'atom_index_1',

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

        'y_0',

        'artg_molecule_type_dist_std_diff',
        'artg_molecule_atom_1_dist_min_diff',
        'artg_molecule_atom_index_0_x_1_std',
        'artg_molecule_dist_min',
        'artg_molecule_atom_index_0_dist_min_diff',
        'artg_molecule_atom_index_0_y_1_mean_diff',
        'artg_molecule_type_dist_min',
        'artg_molecule_atom_1_dist_min_div',

        'atom_index_0',

        'artg_molecule_dist_max',
        'artg_molecule_atom_1_dist_std_diff',
        'artg_molecule_type_dist_max',
        'artg_molecule_atom_index_0_y_1_max_diff',
        'artg_molecule_type_0_dist_std_diff',
        'artg_molecule_type_dist_mean_diff',
        'artg_molecule_atom_1_dist_mean',
        'artg_molecule_atom_index_0_y_1_mean_div',
        'artg_molecule_type_dist_mean_div',

        'type',

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
        'cran_EN_a0', 'cran_rad_a0', 'cran_n_bonds_a0', 'cran_bond_lengths_mean_a0', 'cran_bond_lengths_std_a0',
        'cran_bond_lengths_median_a0',
        'cran_EN_a1', 'cran_rad_a1', 'cran_n_bonds_a1', 'cran_bond_lengths_mean_a1', 'cran_bond_lengths_std_a1',
        'cran_bond_lengths_median_a1',

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

    labels = {}
    for f in ['atom_1', 'type_0', 'type']:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

            labels[f] = lbl

    X = train[good_columns].copy()
    X_test = test[good_columns].copy()

    return X, X_test, labels


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

##### COPY__PASTE__LIB__END #####
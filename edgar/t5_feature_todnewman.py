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
OUTPUT_DIR = '../feature/t5_v2/todn'

SEED = 55
np.random.seed(SEED)


# https://www.kaggle.com/todnewman/keras-neural-net-for-champs
def add_features(df):
    df["distance_center0"] = ((df['x_0'] - df['c_x']) ** 2 + (df['y_0'] - df['c_y']) ** 2 + (
            df['z_0'] - df['c_z']) ** 2) ** (1 / 2)
    df["distance_center1"] = ((df['x_1'] - df['c_x']) ** 2 + (df['y_1'] - df['c_y']) ** 2 + (
            df['z_1'] - df['c_z']) ** 2) ** (1 / 2)
    df["distance_c0"] = ((df['x_0'] - df['x_closest_0']) ** 2 + (df['y_0'] - df['y_closest_0']) ** 2 + (
            df['z_0'] - df['z_closest_0']) ** 2) ** (1 / 2)
    df["distance_c1"] = ((df['x_1'] - df['x_closest_1']) ** 2 + (df['y_1'] - df['y_closest_1']) ** 2 + (
            df['z_1'] - df['z_closest_1']) ** 2) ** (1 / 2)
    df["distance_f0"] = ((df['x_0'] - df['x_farthest_0']) ** 2 + (df['y_0'] - df['y_farthest_0']) ** 2 + (
            df['z_0'] - df['z_farthest_0']) ** 2) ** (1 / 2)
    df["distance_f1"] = ((df['x_1'] - df['x_farthest_1']) ** 2 + (df['y_1'] - df['y_farthest_1']) ** 2 + (
            df['z_1'] - df['z_farthest_1']) ** 2) ** (1 / 2)
    df["vec_center0_x"] = (df['x_0'] - df['c_x']) / (df["distance_center0"] + 1e-10)
    df["vec_center0_y"] = (df['y_0'] - df['c_y']) / (df["distance_center0"] + 1e-10)
    df["vec_center0_z"] = (df['z_0'] - df['c_z']) / (df["distance_center0"] + 1e-10)
    df["vec_center1_x"] = (df['x_1'] - df['c_x']) / (df["distance_center1"] + 1e-10)
    df["vec_center1_y"] = (df['y_1'] - df['c_y']) / (df["distance_center1"] + 1e-10)
    df["vec_center1_z"] = (df['z_1'] - df['c_z']) / (df["distance_center1"] + 1e-10)
    df["vec_c0_x"] = (df['x_0'] - df['x_closest_0']) / (df["distance_c0"] + 1e-10)
    df["vec_c0_y"] = (df['y_0'] - df['y_closest_0']) / (df["distance_c0"] + 1e-10)
    df["vec_c0_z"] = (df['z_0'] - df['z_closest_0']) / (df["distance_c0"] + 1e-10)
    df["vec_c1_x"] = (df['x_1'] - df['x_closest_1']) / (df["distance_c1"] + 1e-10)
    df["vec_c1_y"] = (df['y_1'] - df['y_closest_1']) / (df["distance_c1"] + 1e-10)
    df["vec_c1_z"] = (df['z_1'] - df['z_closest_1']) / (df["distance_c1"] + 1e-10)
    df["vec_f0_x"] = (df['x_0'] - df['x_farthest_0']) / (df["distance_f0"] + 1e-10)
    df["vec_f0_y"] = (df['y_0'] - df['y_farthest_0']) / (df["distance_f0"] + 1e-10)
    df["vec_f0_z"] = (df['z_0'] - df['z_farthest_0']) / (df["distance_f0"] + 1e-10)
    df["vec_f1_x"] = (df['x_1'] - df['x_farthest_1']) / (df["distance_f1"] + 1e-10)
    df["vec_f1_y"] = (df['y_1'] - df['y_farthest_1']) / (df["distance_f1"] + 1e-10)
    df["vec_f1_z"] = (df['z_1'] - df['z_farthest_1']) / (df["distance_f1"] + 1e-10)
    df["vec_x"] = (df['x_1'] - df['x_0']) / df["distance"]
    df["vec_y"] = (df['y_1'] - df['y_0']) / df["distance"]
    df["vec_z"] = (df['z_1'] - df['z_0']) / df["distance"]
    df["cos_c0_c1"] = df["vec_c0_x"] * df["vec_c1_x"] + df["vec_c0_y"] * df["vec_c1_y"] + df["vec_c0_z"] * df[
        "vec_c1_z"]
    df["cos_f0_f1"] = df["vec_f0_x"] * df["vec_f1_x"] + df["vec_f0_y"] * df["vec_f1_y"] + df["vec_f0_z"] * df[
        "vec_f1_z"]
    df["cos_center0_center1"] = df["vec_center0_x"] * df["vec_center1_x"] + df["vec_center0_y"] * df["vec_center1_y"] + \
                                df["vec_center0_z"] * df["vec_center1_z"]
    df["cos_c0"] = df["vec_c0_x"] * df["vec_x"] + df["vec_c0_y"] * df["vec_y"] + df["vec_c0_z"] * df["vec_z"]
    df["cos_c1"] = df["vec_c1_x"] * df["vec_x"] + df["vec_c1_y"] * df["vec_y"] + df["vec_c1_z"] * df["vec_z"]
    df["cos_f0"] = df["vec_f0_x"] * df["vec_x"] + df["vec_f0_y"] * df["vec_y"] + df["vec_f0_z"] * df["vec_z"]
    df["cos_f1"] = df["vec_f1_x"] * df["vec_x"] + df["vec_f1_y"] * df["vec_y"] + df["vec_f1_z"] * df["vec_z"]
    df["cos_center0"] = df["vec_center0_x"] * df["vec_x"] + df["vec_center0_y"] * df["vec_y"] + df["vec_center0_z"] * \
                        df["vec_z"]
    df["cos_center1"] = df["vec_center1_x"] * df["vec_x"] + df["vec_center1_y"] * df["vec_y"] + df["vec_center1_z"] * \
                        df["vec_z"]
    df = df.drop(['vec_c0_x', 'vec_c0_y', 'vec_c0_z', 'vec_c1_x', 'vec_c1_y', 'vec_c1_z',
                  'vec_f0_x', 'vec_f0_y', 'vec_f0_z', 'vec_f1_x', 'vec_f1_y', 'vec_f1_z',
                  'vec_center0_x', 'vec_center0_y', 'vec_center0_z', 'vec_center1_x', 'vec_center1_y', 'vec_center1_z',
                  'vec_x', 'vec_y', 'vec_z'], axis=1)

    cols = ['distance_center0', 'distance_center1', 'distance_c0', 'distance_c1', 'distance_f0', 'distance_f1',
            'cos_c0_c1', 'cos_f0_f1', 'cos_center0_center1', 'cos_c0', 'cos_c1', 'cos_f0', 'cos_f1', 'cos_center0',
            'cos_center1']
    df[cols] = df[cols].astype('float32')


def t5_todnewton_features(train, test):
    for df in [train, test]:
        add_features(df)

train, test, structures, contributions = t5_load_data(INPUT_DIR)
train, test = t5_merge_structures(train, test, structures)
t5_distance_feature(train, test)

structures['c_x'] = structures.groupby('molecule_name')['x'].transform('mean')
structures['c_y'] = structures.groupby('molecule_name')['y'].transform('mean')
structures['c_z'] = structures.groupby('molecule_name')['z'].transform('mean')
structures['atom_n'] = structures.groupby('molecule_name')['atom_index'].transform('max')

train = pd.merge(train, structures[['molecule_name', 'c_x', 'c_y', 'c_z']], how='left', on='molecule_name')
test = pd.merge(test, structures[['molecule_name', 'c_x', 'c_y', 'c_z']], how='left', on='molecule_name')
print(train.dtypes.T)

t5_todnewton_features(train, test)

train.to_csv(f'{OUTPUT_DIR}/todn_train.csv', index=False)
train.to_parquet(f'{OUTPUT_DIR}/todn_train.parquet', index=False)
test.to_csv(f'{OUTPUT_DIR}/todn_test.csv', index=False)
test.to_parquet(f'{OUTPUT_DIR}/todn_test.parquet', index=False)

print(train.shape)
print(train.dtypes.T)

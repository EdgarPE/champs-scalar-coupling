import numpy as np
import pandas as pd
import os
import sys

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t5_lib import *

##### COPY__PASTE__LIB__END #####

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR = '.'
WORK_DIR = '../work/t5'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../feature/t5/edgar'

SEED = 55
np.random.seed(SEED)

bin_size_list = {
    '1JHC': [9, 50, 250],  # ~ 80.000 / subtype
    '2JHC': [14, 75, 375],
    '3JHC': [19, 100, 500],
    '1JHN': [1, 1, 1],
    '2JHN': [2, 10, 50],
    '3JHN': [3, 15, 75],
    # mindegy, hogy 2,3,5 vagy 9! felé osztom. Akkor is -2.15 lesz, van egy rövid rész, ahol gyenge nagyon
    '2JHH': [5, 25, 125],
    '3JHH': [7, 35, 70],
}

columns = {
    'qcut_subtype_0': 'dist_lin',
    'qcut_subtype_1': 'd_2_min',
    'qcut_subtype_2': 'd_3_min',
}

from itertools import tee


def t5_edgar_distmin(df):
    df['d_2_min'] = df[['d_2_0', 'd_2_1']].min(axis=1)
    df['d_3_min'] = df[['d_3_0', 'd_3_1', 'd_3_2']].min(axis=1)


def do_qcut(columns_iter, train, test, train_row_sel, test_row_sel, type):
    try:
        qcut_col_idx, qcut_col = next(columns_iter)
    except StopIteration:
        return train, test

    print(qcut_col_idx, qcut_col)
    data_col = columns[qcut_col]
    bin_size = bin_size_list[type][qcut_col_idx]

    print(f'type: {type}, qcut_level: {qcut_col_idx}, bin_size: {bin_size}')

    data = list(train[train_row_sel][data_col]) + list(test[test_row_sel][data_col])
    __ser, bins = pd.qcut(data, bin_size, retbins=True, labels=False)

    train.loc[train_row_sel, qcut_col] = pd.cut(train.loc[train_row_sel, data_col],
                                                bins=bins, labels=False,
                                                include_lowest=True).astype('category')
    test.loc[test_row_sel, qcut_col] = pd.cut(test.loc[test_row_sel, data_col],
                                              bins=bins, labels=False,
                                              include_lowest=True).astype('category')

    for b in range(bin_size):
        columns_iter, sub_iter = tee(columns_iter)
        train_row_sel = train_row_sel & (bins[b] <= train[qcut_col]) & (train[qcut_col] <= bins[(b + 1)])
        test_row_sel = test_row_sel & (bins[b] <= test[qcut_col]) & (test[qcut_col] <= bins[(b + 1)])

        train, test = do_qcut(sub_iter, train, test, train_row_sel, test_row_sel, type)

    return train, test


def t5_edgar_qcut_nd(train, test):
    t5_edgar_distmin(train)
    t5_edgar_distmin(test)

    # print(train[['d_2_0', 'd_2_1', 'd_2_min']].head())
    # print(train[['d_3_0', 'd_3_1', 'd_3_2', 'd_3_min']].head(200))
    # exit(0)

    for qcut_col in columns:
        train[qcut_col] = test[qcut_col] = -1

    for t in train['type'].unique():
        train_row_sel = train['type'] ==  t
        test_row_sel = test['type'] == t
        columns_iter = enumerate(columns)

        do_qcut(columns_iter, train, test, train_row_sel, test_row_sel, t)

    return train[columns].astype('category'), test[columns].astype('category')


# Dist feature
train, test, structures, contributions = t5_read_parquet(WORK_DIR)





train_, test_ = t5_edgar_qcut_nd(train, test)

train_.to_csv(f'{OUTPUT_DIR}/edgar_train.csv', index=False)
train_.to_parquet(f'{OUTPUT_DIR}/edgar_train.parquet', index=False)
test_.to_csv(f'{OUTPUT_DIR}/edgar_test.csv', index=False)
test_.to_parquet(f'{OUTPUT_DIR}/edgar_test.parquet', index=False)

print(train_.shape)
print(train_.dtypes.T)

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

# INPUT_DIR = '../input'
INPUT_DIR = '../work/subsample_5000'

# WORK_DIR = '.'
WORK_DIR = '../work/t5'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../feature/t5/edgar'

SEED = 55
np.random.seed(SEED)


def t5_edgar_qcut(train, test):
    keep = []
    for bin_size in [5, 21, 127]:
        train[f'dist_qcut_{bin_size}'] = -1
        test[f'dist_qcut_{bin_size}'] = -1
        keep.append(f'dist_qcut_{bin_size}')

        for t in train['type'].unique():
            print(f'type: {t}, bin_size: {bin_size}')
            data = list(train[train['type'] == t]['dist_lin']) + list(test[test['type'] == t]['dist_lin'])

            _ser, bins = pd.qcut(data, bin_size, retbins=True, labels=False)

            train.loc[train['type'] == t, f'dist_qcut_{bin_size}'] = pd.cut(train.loc[train['type'] == t, 'dist_lin'],
                                                                            bins=bins, labels=False,
                                                                            include_lowest=True).astype('int8')
            test.loc[test['type'] == t, f'dist_qcut_{bin_size}'] = pd.cut(test.loc[test['type'] == t, 'dist_lin'],
                                                                          bins=bins, labels=False,
                                                                          include_lowest=True).astype('int8')
    return train[keep].astype('int8'), test[keep].astype('int8')


# Dist feature
train, test, structures, contributions = t5_read_parquet(WORK_DIR)

train_, test_ = t5_edgar_qcut(train, test)

train_.to_csv(f'{OUTPUT_DIR}/edgar_train.csv', index=False)
train_.to_parquet(f'{OUTPUT_DIR}/edgar_train.parquet', index=False)
test_.to_csv(f'{OUTPUT_DIR}/edgar_test.csv', index=False)
test_.to_parquet(f'{OUTPUT_DIR}/edgar_test.parquet', index=False)

print(train_.shape)
print(train_.dtypes.T)

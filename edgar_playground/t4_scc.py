import numpy as np
import pandas as pd
import os
import sys

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work/t4'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t4'

TYPE_WL = ['1JHC','2JHC','3JHC','1JHN','2JHN','3JHN','2JHH','3JHH']
# TYPE_WL = ['1JHC']

TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 3,
    # '1JHC': 5,
}

N_ESTIMATORS = {
    '_': 1000,
    # '1JHC': 20000,
}

PARAMS = {
    '_': {
        'num_leaves': 128,
        'min_child_samples': 9,
        'objective': 'regression',
        'max_depth': 9,
        'learning_rate': 0.1,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": SEED,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'colsample_bytree': 1.0
    },
    '1JHN': {'subsample': 1, 'learning_rate': 0.05},
    '2JHN': {'subsample': 1, 'learning_rate': 0.05},
    '3JHN': {'subsample': 1, 'learning_rate': 0.05},
}
##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)

from edgar_playground.t4_lib import train_model_regression
from edgar_playground.t4_lib import t4_load_data
from edgar_playground.t4_lib import t4_load_data_mulliken
from edgar_playground.t4_lib import t4_load_data_mulliken_oof
from edgar_playground.t4_lib import t4_load_data_contributions_oof
from edgar_playground.t4_lib import t4_preprocess_data
from edgar_playground.t4_lib import t4_preprocess_data_mulliken
from edgar_playground.t4_lib import t4_create_features
from edgar_playground.t4_lib import t4_prepare_columns
from edgar_playground.t4_lib import t4_to_parquet
from edgar_playground.t4_lib import t4_read_parquet
from edgar_playground.t4_lib import t4_do_predict

##### COPY__PASTE__LIB__END #####

train, test, structures, contributions = t4_load_data(INPUT_DIR)

train, test = t4_load_data_mulliken_oof(WORK_DIR, train, test)

train, test, structures = t4_preprocess_data(train, test, structures, contributions)

train, test = t4_load_data_contributions_oof(WORK_DIR, train, test)

t4_create_features(train, test)

X, X_test, labels = t4_prepare_columns(train, test,
                                          good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1', 'fc', 'sd',
                                                              'pso', 'dso'])

t4_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels)

train[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t4_scc_train.csv', index=False)
test.rename(inplace=True, columns={'oof_scalar_coupling_constant': 'scalar_coupling_constant',})
test[['id'] + [f'{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t4_scc_test.csv', index=False)
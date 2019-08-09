import numpy as np
import pandas as pd
import os
import sys

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t4_lib import *

##### COPY__PASTE__LIB__END #####

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work/t4_play'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t4_play'

TYPE_WL = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
# TYPE_WL = ['1JHC']

TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 3,
}

N_ESTIMATORS = {
    '_': 1000
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

train, test, structures, contributions = t4_load_data(INPUT_DIR)

structures = t4_merge_yukawa(INPUT_DIR, structures)

structures = t4_crane_features(structures)

train, test = t4_merge_structures(train, test, structures)

t4_distance_feature(train, test)

t4_artgor_features(train, test)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t4_load_data_mulliken_oof(WORK_DIR, train, test)

#
# Load Phase 2. OOF data Contributions (fc, sd, pso, dso)
#
train, test = t4_load_data_contributions_oof(WORK_DIR, train, test)

#
# Predict final target (Scalar coupling constant)
#
X, X_test, labels = t4_prepare_columns(train, test,
                                       good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1', 'fc', 'sd',
                                                           'pso', 'dso'])
t4_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels)

train[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t4c_scc_train.csv', index=False)
test.rename(inplace=True, columns={'oof_scalar_coupling_constant': 'scalar_coupling_constant'})
test[['id'] + [f'{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t4c_scc_test.csv', index=False)

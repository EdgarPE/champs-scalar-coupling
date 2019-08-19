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

# WORK_DIR= '.'
WORK_DIR = '../work/t5'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t5'

TYPE_WL = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
# TYPE_WL = ['1JHN']

TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 5,
}

N_ESTIMATORS = {
    '_': 12000,
    # '1JHC': 6000,
    # '2JHC': 4000,
    # '3JHC': 6000,
    # '1JHN': 6000,
}

PARAMS = {
    '_': {
        'num_leaves': 128,
        'min_child_samples': 22,
        'objective': 'regression',
        'max_depth': 9,
        'learning_rate': 0.2,
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
    # '1JHN': {'subsample': 1, 'learning_rate': 0.05, 'min_child_samples': 5, 'num_leaves': 500, 'max_depth': 11},
    '2JHN': {'subsample': 1, 'learning_rate': 0.05},
    '3JHN': {'subsample': 1, 'learning_rate': 0.05},
    '1JHC': {'min_child_samples': 120},
    # '2JHC': {'min_child_samples': 500, 'learning_rate': 0.2, 'num_leaves': 500, 'max_depth': 11},

}

# train, test, structures, contributions = t5_load_data(INPUT_DIR)
#
# train, test = t5_load_feature_criskiev(FEATURE_DIR, train, test)
#
# structures = t5_merge_yukawa(INPUT_DIR, structures)
#
# structures = t5_load_feature_crane(FEATURE_DIR, structures)
#
# train, test = t5_merge_structures(train, test, structures)
#
# t5_distance_feature(train, test)
#
# train, test = t5_load_feature_artgor(FEATURE_DIR, train, test)

#
# Save to and/or load from parquet
#
# t5_to_parquet(WORK_DIR, train, test, structures, contributions)

train, test, structures, contributions = t5_read_parquet(WORK_DIR)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t5_load_data_mulliken_oof(WORK_DIR, train, test)

#
# Load Phase 2. OOF data Contributions (fc, sd, pso, dso)
#
train, test = t5_load_data_contributions_oof(WORK_DIR, train, test)

# t5_criskiev_features_extra(train, test)

#
# Predict final target (Scalar coupling constant)
#

# pd.set_option('display.max_rows', 200)
# print(train.describe().T) # Verbose=True
# print(train.dtypes.T)

X, X_test, labels = t5_prepare_columns(train, test,
                                       good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1', 'fc', 'sd',
                                                           'pso', 'dso', 'contrib_sum'])
t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels)

train[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t5c_scc_train.csv', index=False)
test.rename(inplace=True, columns={'oof_scalar_coupling_constant': 'scalar_coupling_constant'})
test[['id'] + [f'{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t5c_scc_test.csv', index=False)

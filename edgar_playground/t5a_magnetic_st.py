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

# FEATURE_DIR = '.'
FEATURE_DIR = '../feature/t5'

# WORK_DIR= '.'
WORK_DIR = '../work/t5'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t5'

TYPE_WL = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
# TYPE_WL = ['1JHC']

TARGET_WL = ['XX', 'YY', 'ZZ']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 5,
}

N_ESTIMATORS = {'_': 10000}

PARAMS = {
    '_': {
        'num_leaves': 128,
        'min_child_samples': 22,
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
    '1JHC': {'min_child_samples': 120},
}

train, test, structures, contributions = t5_load_data(INPUT_DIR)

train, test = t5_load_feature_criskiev(FEATURE_DIR, train, test)

structures = t5_merge_yukawa(INPUT_DIR, structures)

structures = t5_load_feature_crane(FEATURE_DIR, structures)

train, test = t5_merge_structures(train, test, structures)

t5_distance_feature(train, test)

train, test = t5_load_feature_artgor(FEATURE_DIR, train, test)

#
# Save to and/or load from parquet
#
# t5_to_parquet(WORK_DIR, train, test, structures, contributions)

# train, test, structures, contributions = t5_read_parquet(WORK_DIR)

#
# Edike :)
#
train, test = t5_load_feature_edgar(FEATURE_DIR, train, test)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t5_load_data_mulliken_oof(WORK_DIR, train, test)

#
# Predict Magnetic S.T.
#
X, X_test, labels = t5_prepare_columns(train, test, good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1'])
t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels)

#
# Predict Mulliken charge
#
X, X_test, labels = t5_prepare_columns(train, test)
t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels)

train = train[['molecule_name', 'atom_index_0', 'oof_mulliken_charge']].rename(columns={'atom_index_0': 'atom_index'})
mean = train.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].mean()
mean.to_csv(f'{OUTPUT_DIR}/t5a_mulliken_train.csv', index=True)

test = test[['molecule_name', 'atom_index_0', 'oof_mulliken_charge']].rename(columns={'atom_index_0': 'atom_index'})
mean = test.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].mean()
mean.to_csv(f'{OUTPUT_DIR}/t5a_mulliken_test.csv', index=True)

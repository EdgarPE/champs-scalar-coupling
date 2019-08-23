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

# TYPE_WL = ['2JHH', '3JHH', '1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', ]
TYPE_WL = ['1JHC', '2JHC', '3JHC', '2JHH', '3JHH', '1JHN', '2JHN', '3JHN',]

# TARGET_WL = ['fc', 'sd', 'pso', 'dso']
TARGET_WL = ['fc']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 5, # mint UA
}

N_ESTIMATORS = {
    '_': 12000, # 8000-nek még van értelme fc-re
}

PARAMS = {
    '_': {
        'num_leaves': 512,
        'min_child_samples': 12,
        'objective': 'regression',
        'max_depth': 12,
        'learning_rate': 0.02,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": SEED,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.2,
        'reg_lambda': 0.3,
        'colsample_bytree': 0.7
    },
    '1JHN': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05},
    # '2JHN': {'subsample': 1, 'learning_rate': 0.02},
    # '3JHN': {'subsample': 1, 'learning_rate': 0.02},
    # '1JHC': {'min_child_samples': 22},
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
# Edike :)
#
train, test = t5_load_feature_edgar(FEATURE_DIR, train, test)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t5_load_data_mulliken_oof(WORK_DIR, train, test)

#
# Merge contributions fact data
#
train = t5_merge_contributions(train, contributions)

#
# Predict contributions
#
X, X_test, labels = t5_prepare_columns(train, test, good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1',
                                                                        'qcut_subtype_0', 'qcut_subtype_1', 'qcut_subtype_2'])
t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels, OUTPUT_DIR,
              't5b_contributions_train.csv', 't5b_contributions_test.csv')

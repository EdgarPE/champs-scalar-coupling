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

# TYPE_WL = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
TYPE_WL = ['3JHC', '2JHN']

TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 5, # mint UA
}

N_ESTIMATORS = {
    '_': 12000,
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
        'reg_alpha': 0.01,
        'reg_lambda': 0.05,
        'colsample_bytree': 0.4
    },
    # '1JHN': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05, },
    # '2JHN': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05, },
    # '3JHN': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05, },
    # '1JHC': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05, },
    # '3JHC': {'colsample_bytree': 0.4, 'reg_alpha': 0.01, 'reg_lambda': 0.05, },
}

# train, test, structures, contributions = t5_load_data(INPUT_DIR)
# gc.collect()
# disp_mem_usage()
#
# structures = t5_merge_yukawa(INPUT_DIR, structures)
# gc.collect()
# disp_mem_usage()
#
# structures = t5_merge_qm7eigen(FEATURE_DIR, structures)
# gc.collect()
# disp_mem_usage()
#
# structures = t5_load_feature_crane(FEATURE_DIR, structures)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_merge_structures(train, test, structures)
# gc.collect()
# disp_mem_usage()
#
# # print(structures.dtypes.T)
# del structures
# gc.collect()
#
# t5_distance_feature(train, test)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_feature_criskiev(FEATURE_DIR, train, test)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_feature_artgor(FEATURE_DIR, train, test)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_feature_giba(FEATURE_DIR, train, test)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_feature_edgar(FEATURE_DIR, train, test)
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_data_mulliken_oof(WORK_DIR, train, test)  # Phase 1. OOF data Mulliken charge
# gc.collect()
# disp_mem_usage()
#
# train, test = t5_load_data_contributions_oof(WORK_DIR, train, test)  # Phase 2. OOF Contributions (fc, sd, pso, dso)
# gc.collect()
# disp_mem_usage()

#
# Save to and/or load from parquet
#
# t5_to_parquet_tt(WORK_DIR, train, test)
train, test, = t5_read_parquet_tt(WORK_DIR)
gc.collect()
disp_mem_usage()

#
# Predict final target (Scalar coupling constant)
#

# pd.set_option('display.max_rows', 200)
# print(train.describe().T) # Verbose=True
# print(train.dtypes.T)

extra_cols = []
extra_cols += ['mulliken_charge_0', 'mulliken_charge_1']
extra_cols += ['fc', 'sd', 'pso', 'dso', 'contrib_sum']
extra_cols += ['qcut_subtype_0', 'qcut_subtype_1', 'qcut_subtype_2']
X, X_test, labels = t5_prepare_columns(train, test, good_columns_extra=extra_cols)
t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels, OUTPUT_DIR,
              't5c_scc_train.csv', 't5c_scc_test.csv')

test.rename(inplace=True, columns={'oof_scalar_coupling_constant': 'scalar_coupling_constant'})
test[['id'] + [f'{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t5c_submission.csv', index=False)

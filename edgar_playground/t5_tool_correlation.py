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

# TARGET_WL = ['fc', 'sd', 'pso', 'dso']

SEED = 55
np.random.seed(SEED)


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
# train, test = t5_load_feature_giba(FEATURE_DIR, train, test)

#
# Save to and/or load from parquet
#
# t5_to_parquet(WORK_DIR, train, test, structures, contributions)

train, test, structures, contributions = t5_read_parquet(WORK_DIR)
disp_mem_usage()
print(train.shape)
print(test.shape)

desc = train.describe()
desc.to_csv(f'{WORK_DIR}/t5_describe_train.csv', index=False)
desc.to_parquet(f'{WORK_DIR}/t5_describe_train.csv', index=False)

desc = test.describe()
desc.to_csv(f'{WORK_DIR}/t5_describe_test.csv', index=False)
desc.to_parquet(f'{WORK_DIR}/t5_describe_test.parquet', index=False)

# corr = train.corr()
# corr.to_csv(f'{WORK_DIR}/t5_correlation_train.csv', index=False)
# corr.to_parquet(f'{WORK_DIR}/t5_correlation_train.parquet', index=False)
#
# corr = test.corr()
# corr.to_csv(f'{WORK_DIR}/t5_correlation_test.csv', index=False)
# corr.to_parquet(f'{WORK_DIR}/t5_correlation_test.parquet', index=False)

train.fillna(-10000)
test.fillna(-10000)

corr = train.corr()
corr.to_csv(f'{WORK_DIR}/t5_correlation_fillna_train.csv', index=False)
corr.to_parquet(f'{WORK_DIR}/t5_correlation_fillna_train.parquet', index=False)

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
print(to_drop)

corr = test.corr()
corr.to_csv(f'{WORK_DIR}/t5_correlation_fillna_test.csv', index=False)
corr.to_parquet(f'{WORK_DIR}/t5_correlation_fillna_test.parquet', index=False)

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
print(to_drop)

exit(0)

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
for c in train.columns: print(c)

extra_cols = []
extra_cols += ['mulliken_charge_0', 'mulliken_charge_1']
extra_cols += ['qcut_subtype_0', 'qcut_subtype_1', 'qcut_subtype_2']
X, X_test, labels = t5_prepare_columns(train, test, good_columns_extra=extra_cols)
# t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels, OUTPUT_DIR,
#               't5b_contributions_train.csv', 't5b_contributions_test.csv')

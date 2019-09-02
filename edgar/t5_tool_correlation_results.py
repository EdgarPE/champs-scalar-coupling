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

corr_matrix = pd.read_csv(f'{WORK_DIR}/t5_correlation_fillna_train.csv')

"""
Method 1 (shit)
"""
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# # Find index of feature columns with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.99999)]


"""
Method 2
"""
threshold = 1 - 1e-06
to_drop = set() # Set of all the names of deleted columns
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in to_drop):
            colname = corr_matrix.columns[i] # getting the name of column
            print((corr_matrix.columns[i], corr_matrix.columns[j]))
            to_drop.add(colname)

# print(to_drop)

# print([c for c in corr_matrix.columns if c[0:5] == 'giba_'])

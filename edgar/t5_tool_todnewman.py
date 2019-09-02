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

FEATURE_DIR = '../feature/t5/todnewman'

SEED = 55
np.random.seed(SEED)

cols = ['distance_center0', 'distance_center1', 'distance_c0', 'distance_c1', 'distance_f0', 'distance_f1',
        'cos_c0_c1', 'cos_f0_f1', 'cos_center0_center1', 'cos_c0', 'cos_c1', 'cos_f0', 'cos_f1', 'cos_center0',
        'cos_center1']
# df[cols] = df[cols].astype('float32')

train = pd.read_parquet(f'{FEATURE_DIR}/df_train.parquet')[cols]
test = pd.read_parquet(f'{FEATURE_DIR}/df_test.parquet')[cols]

rename = {c: f'todn_{c}' for c in cols}
train.rename(inplace=True, columns=rename)
test.rename(inplace=True, columns=rename)

train.to_parquet(f'{FEATURE_DIR}/todnewman_train.parquet', index=False)
train.to_csv(f'{FEATURE_DIR}/todnewman_train.csv', index=False)

print(train.shape)
print(train.dtypes.T)

test.to_parquet(f'{FEATURE_DIR}/todnewman_test.parquet', index=False)
test.to_csv(f'{FEATURE_DIR}/todnewman_test.csv', index=False)

print(test.shape)
print(test.dtypes.T)


import numpy as np
import pandas as pd
import os
import sys

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)

from time import time
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t5_lib import *

##### COPY__PASTE__LIB__END #####

# INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR = '.'
WORK_DIR = '../work/t5'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../feature/t5/edgar'

SEED = 55
np.random.seed(SEED)

def t5_edgar_kmeans(train, test):
    bin_size_map = {
        '1JHC': [9, 50, 250], # ~ 80.000 / subtype
        '2JHC': [14, 75, 375],
        '3JHC': [19, 100, 500],
        '1JHN': [1, 1, 1],
        '2JHN': [2, 10, 50],
        '3JHN': [3, 15, 75], # mindegy, hogy 2,3,5 vagy 9! felé osztom. Akkor is -2.15 lesz, van egy rövid rész, ahol gyenge nagyon
        '2JHH': [5, 25, 125],
        '3JHH': [7, 35, 70],
    }

    columns = ['kmeans_subtype_0', 'kmeans_subtype_1', 'kmeans_subtype_2']

    for c in columns:
        train[c] = test[c] = -1

    for t in train['type'].unique():
        for col_index, bin_size in enumerate(bin_size_map[t]):
            print(f'type: {t}, bin_size: {bin_size}')

            data = train.to_numpy()

            # reduced_data = PCA(n_components=bin_size).fit(data)
            kmeans = KMeans(init='k-means++', n_clusters=bin_size, n_init=10)
            kmeans.fit(data)


    return train[columns].astype('int8'), test[columns].astype('int8')


# Dist feature
train, test, structures, contributions = t5_read_parquet(WORK_DIR)


train_, test_ = t5_edgar_kmeans(train, test)

train_.to_csv(f'{OUTPUT_DIR}/edgar_train.csv', index=False)
train_.to_parquet(f'{OUTPUT_DIR}/edgar_train.parquet', index=False)
test_.to_csv(f'{OUTPUT_DIR}/edgar_test.csv', index=False)
test_.to_parquet(f'{OUTPUT_DIR}/edgar_test.parquet', index=False)

print(train_.shape)
print(train_.dtypes.T)

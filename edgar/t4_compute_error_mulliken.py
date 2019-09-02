import numpy as np
import pandas as pd
import math

def mean_log_mae(y_true, y_pred):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    return math.log((y_true-y_pred).abs().mean())


y_true = pd.read_csv('../input/mulliken_charges.csv')
y_pred = pd.read_csv('../work/t3_mull_v2_train.csv')

t = 'mulliken_charge'
print('Mean  : %.5f' % mean_log_mae(y_true[t], y_pred[f'oof_{t}_mean']))
print('Median: %.5f' % mean_log_mae(y_true[t], y_pred[f'oof_{t}_median']))

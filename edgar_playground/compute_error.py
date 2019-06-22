import numpy as np
import pandas as pd

y_pred = pd.read_csv('../work/t0_oof_train.csv')
y_true = pd.read_csv('../input/train.csv')

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


print('%.5f' % group_mean_log_mae(y_true['scalar_coupling_constant'], y_pred['scalar_coupling_constant'], y_true['type']))

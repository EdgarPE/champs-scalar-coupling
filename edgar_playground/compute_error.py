import numpy as np
import pandas as pd


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    print(maes)
    print(np.log(maes.map(lambda x: max(x, floor))))
    return np.log(maes.map(lambda x: max(x, floor))).mean()


# y_true = pd.read_csv('../input/train.csv')
# y_pred = pd.read_csv('../work/t0_oof_train.csv')
# print('%.5f' % group_mean_log_mae(y_true['scalar_coupling_constant'], y_pred['oof_scalar_coupling_constant'], y_true['type']))


y_true = pd.read_csv('../input/scalar_coupling_contributions.csv')
y_pred = pd.read_csv('../work/t0_oof_fc_train.csv')
print('%.5f' % group_mean_log_mae(y_true['fc'], y_pred['oof_fc'], y_true['type']))

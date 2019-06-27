import numpy as np
import pandas as pd


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Típusonként adom vissza, nem egyben

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))) # .mean()


# y_true = pd.read_csv('../input/train.csv')
# y_pred = pd.read_csv('../work/t0_oof_train.csv')
# print('%.5f' % group_mean_log_mae(y_true['scalar_coupling_constant'], y_pred['oof_scalar_coupling_constant'], y_true['type']))

types = ['1JHN','2JHN','3JHN','2JHH','3JHH','1JHC','2JHC','3JHC']

err = pd.DataFrame({}, index=types)

y_true = pd.read_csv('../input/scalar_coupling_contributions.csv')
y_pred = pd.read_csv('../work/t1_baseline_train.csv')

for t in ['fc', 'sd', 'pso', 'dso']:
    err[t] = ['%.3f' % e for e in group_mean_log_mae(y_true[t], y_pred[f'oof_{t}'], y_true['type'])]

print(err)

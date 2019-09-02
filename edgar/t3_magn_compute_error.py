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


y_pred = pd.read_csv('../work/t3_magn_train_500.csv')

train  = pd.read_csv('../input/train.csv')
magnetic_st = pd.read_csv('../input/magnetic_shielding_tensors.csv')

train = pd.merge(train, magnetic_st, how='left',
                 left_on=['molecule_name', 'atom_index_0'],
                 right_on=['molecule_name', 'atom_index'])
train.drop('atom_index', axis=1, inplace=True)
train.rename(inplace=True, columns={
    'XX': 'magnetic_st_0_XX',
    'XY': 'magnetic_st_0_XY',
    'XZ': 'magnetic_st_0_XZ',
    'YX': 'magnetic_st_0_YX',
    'YY': 'magnetic_st_0_YY',
    'YZ': 'magnetic_st_0_YZ',
    'ZX': 'magnetic_st_0_ZX',
    'ZY': 'magnetic_st_0_ZY',
    'ZZ': 'magnetic_st_0_ZZ',
})

types = ['1JHN','2JHN','3JHN','2JHH','3JHH','1JHC','2JHC','3JHC']

targets = [
    'magnetic_st_0_XX',
    'magnetic_st_0_XY',
    'magnetic_st_0_XZ',
    'magnetic_st_0_YX',
    'magnetic_st_0_YY',
    'magnetic_st_0_YZ',
    'magnetic_st_0_ZX',
    'magnetic_st_0_ZY',
    'magnetic_st_0_ZZ',
    'magnetic_st_1_XX',
    'magnetic_st_1_XY',
    'magnetic_st_1_XZ',
    'magnetic_st_1_YX',
    'magnetic_st_1_YY',
    'magnetic_st_1_YZ',
    'magnetic_st_1_ZX',
    'magnetic_st_1_ZY',
    'magnetic_st_1_ZZ',
]

err = pd.DataFrame({}, index=types)
err_print = err.copy()
for t in targets:
    err[t] = group_mean_log_mae(train[t], y_pred[f'oof_{t}'], train['type'])
    err_print[t] = ['%.3f' % e for e in err[t]]

err_print.loc['mean'] = ['%.3f' % e for e in err.mean(axis=0)]
print(err_print)

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


y_pred = pd.read_csv('../work/t3_mull_train_500.csv')

train  = pd.read_csv('../input/train.csv')
mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')

train = pd.merge(train, mulliken_charges, how='left',
                 left_on=['molecule_name', 'atom_index_0'],
                 right_on=['molecule_name', 'atom_index'])
train.drop('atom_index', axis=1, inplace=True)
train.rename(inplace=True, columns={'mulliken_charge': 'mulliken_charge_0'})

train = pd.merge(train, mulliken_charges, how='left',
                 left_on=['molecule_name', 'atom_index_1'],
                 right_on=['molecule_name', 'atom_index'])
train.drop('atom_index', axis=1, inplace=True)
train.rename(inplace=True, columns={'mulliken_charge': 'mulliken_charge_1'})


types = ['1JHN','2JHN','3JHN','2JHH','3JHH','1JHC','2JHC','3JHC']
targets = ['mulliken_charge_0', 'mulliken_charge_1']

err = pd.DataFrame({}, index=types)
err_print = err.copy()

for t in targets:
    err[t] = group_mean_log_mae(train[t], y_pred[f'oof_{t}'], train['type'])
    err_print[t] = ['%.3f' % e for e in err[t]]

err_print.loc['mean'] = ['%.3f' % e for e in err.mean(axis=0)]
print(err_print)

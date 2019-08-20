#!/usr/bin/env python

import sys
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


y_true = pd.read_csv(sys.argv[1])
y_pred = pd.read_csv(sys.argv[3])

maes = group_mean_log_mae(y_true[sys.argv[2]], y_pred[sys.argv[4]], y_true[sys.argv[5] if len(sys.argv) >= 6 else 'type'])
print(maes)
print('Mean: %.6f' % maes.mean())

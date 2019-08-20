#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

def mean_log_mae(y_true, y_pred):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    return np.log((y_true - y_pred).abs().mean())


m0 = pd.read_csv(sys.argv[1])
m1 = pd.read_csv(sys.argv[2])
merged = pd.concat([m0, m1], axis=0)

y_true = pd.read_csv(sys.argv[3])

full = pd.merge(merged, y_true, how='left',
                  left_on=['molecule_name', 'atom_index'],
                  right_on=['molecule_name', 'atom_index'])

mean = mean_log_mae(full['mulliken_charge'], full['oof_mulliken_charge'])
print('Mean: %.6f' % mean)

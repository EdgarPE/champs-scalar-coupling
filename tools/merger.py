#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

left_file = sys.argv[1]
left_column = sys.argv[2]
right_file = sys.argv[3]
right_column = sys.argv[4]
dataset_file = sys.argv[5] if len(sys.argv) >= 6 else None
type = sys.argv[6] if len(sys.argv) >= 7 else None

left = pd.read_csv(left_file)
right = pd.read_csv(right_file)

if left.shape[0] != right.shape[0]:
    raise Exception('Not same amount of rows')

if dataset_file is None and type is None:
    left[left_column] = right[right_column]
else:
    if dataset_file == 'train':
        dataset = pd.read_csv('/home/edgarpe/kaggle/champs-scalar-coupling/input/train.csv')
    elif dataset_file == 'test':
        dataset = pd.read_csv('/home/edgarpe/kaggle/champs-scalar-coupling/input/test.csv')
    else:
        raise Exception('Dataset unknown')

    left.loc[dataset['type'] == type, left_column] = right.loc[dataset['type'] == type, right_column]

left.to_csv(sys.stdout, index=False)



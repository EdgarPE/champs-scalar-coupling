import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

base_dir = '/home/edgar/uoa/champs-submissions'
output_file = '../work/submission_merger.csv'

scc = 'scalar_coupling_constant'

rename_col = {f'oof_{scc}': scc}


# submission.csv
submission_csv = pd.read_csv('../input/sample_submission.csv', index_col='id')

# UA  | LB -1.721 | CV -1.529
ua = pd.read_csv(f'{base_dir}/t4_ua/t4c_scc_test.csv', index_col='id')


# hetfo | LB -1.486 | CV -1.718
hetfo = pd.read_csv(f'{base_dir}/t5_hetfo/t5c_scc_test.csv', index_col='id')
hetfo.rename(columns=rename_col, inplace=True)


# kedd | LB -1.514 | CV -1.741
kedd = pd.read_csv(f'{base_dir}/t5_v2_kedd/t5_scc_v2_test.csv', index_col='id')
kedd.rename(columns=rename_col, inplace=True)


# szerda | LB -1.516 | CV -1.761
szerda = pd.read_csv(f'{base_dir}/t5_v2_szerda/t5_scc_v2_submission.csv', index_col='id')


# kernel1 | LB -1.717 | CV -1.658
kernel1 = pd.read_csv(f'{base_dir}/kernel1/t6c_scc_kernel_1_717.csv', index_col='id')


# kernel2 | LB -1.341
kernel2 = pd.read_csv(f'{base_dir}/kernel2/t6c_scc_kernel2_1_341.csv', index_col='id')

# #############################################

# LGBM Chriskiev + HPo | LB -1.667 | CV -1.549
# https://www.kaggle.com/filemide/distance-criskiev-hyparam-cont-1-66?scriptVersionId=19166590
filemide = pd.read_csv(f'{base_dir}/public_filemide/submission.csv', index_col='id')

# Criskiev's distances, more estimators, GroupKFold | LB -1.618 | CV -1.477
# https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold?scriptVersionId=18843561
marcogorelli = pd.read_csv(f'{base_dir}/public_marcogorelli/submission.csv', index_col='id')

# Chriskiev fork | LB -1.643 | CV -1.493
# https://www.kaggle.com/harshit92/fork-from-kernel-1-481
harshit92 = pd.read_csv(f'{base_dir}/public_harshit92/submission.csv', index_col='id')

# Keras Neural Net for CHAMPS | LB -1.073 | CV -1.081
# https://www.kaggle.com/todnewman/keras-neural-net-for-champs
# todnewman = pd.read_csv(f'{base_dir}/public_todnewman/workingsubmission-test.csv', index_col='id')

# TodNewman Neural net for champs + Criskiev features | LB -1.672 | CV -1.70
# https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features
xwxw2929 = pd.read_csv(f'{base_dir}/public_xwxw2929/submission.csv', index_col='id')

# "Keras Neural Net and Distance Features" | LB -1.674 | CV -1.70
# https://www.kaggle.com/yamqwe/deep-learning-fork-and-tweaks-lb-1-674
yamqwe = pd.read_csv(f'{base_dir}/public_yamqwe/submission.csv', index_col='id')

# 1. MPNN | LB -1.286
# https://www.kaggle.com/fnands/1-mpnn?scriptVersionId=18233432
fnands = pd.read_csv(f'{base_dir}/public_fnands/submission.csv', index_col='id')

# SchNET | LB -1.327
# https://www.kaggle.com/toshik/schnet-starter-kit?scriptVersionId=19040743
toshik = pd.read_csv(f'{base_dir}/public_toshik/kernel_schnet.csv', index_col='id')

# seed55
seed55 = pd.read_csv(f'./t5_v2_seed55/t5_scc_v2_train.csv', index_col='id')
seed55.rename(columns=rename_col, inplace=True)

seed200 = pd.read_csv(f'./t5_v2_seed200/t5_scc_v2_train.csv', index_col='id')
seed200.rename(columns=rename_col, inplace=True)

# Things to merge
outs = [ua, kedd, kernel1, yamqwe, fnands, toshik]


# result = pd.concat([out['scalar_coupling_constant'] for out in outs], axis=1)
result = pd.concat(outs, axis=1).median(axis=1)

result = result.reset_index()
result.columns = ['id', scc]
print(result)
result.to_csv(output_file, index=False)

# submission_csv[scc] = result
# submission_csv.to_csv(output_file, index=True)

submission_csv.to_csv(output_file, index=True)
print(output_file)

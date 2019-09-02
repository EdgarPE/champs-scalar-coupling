import numpy as np
import pandas as pd

input_dir = '../work'
output_dir = '../work'

fc = pd.read_csv(input_dir + '/t0_oof_fc_train.csv')
sd = pd.read_csv(input_dir + '/t0_oof_sd_train.csv')
pso = pd.read_csv(input_dir + '/t0_oof_pso_train.csv')
dso = pd.read_csv(input_dir + '/t0_oof_dso_train.csv')

pred = fc.copy()
pred['oof_sd'] = sd['oof_sd']
pred['oof_pso'] = pso['oof_pso']
pred['oof_dso'] = dso['oof_dso']

pred['scalar_coupling_constant'] = pred['oof_fc'] + pred['oof_sd'] + pred['oof_pso'] + pred['oof_dso']
pred = pred[['id', 'scalar_coupling_constant']]
pred.to_csv(output_dir + '/t0_oof_train.csv', index=False)
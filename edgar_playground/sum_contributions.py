import numpy as np
import pandas as pd

input_dir = '../work'
output_dir = '../work'

fc = pd.read_csv(input_dir + '/t0_predict_fc.csv')
sd = pd.read_csv(input_dir + '/t0_predict_sd.csv')
pso = pd.read_csv(input_dir + '/t0_predict_pso.csv')
dso = pd.read_csv(input_dir + '/t0_predict_dso.csv')

pred = fc.copy()
pred['pred_sd'] = sd['pred_sd']
pred['pred_pso'] = pso['pred_pso']
pred['pred_dso'] = dso['pred_dso']

pred['scalar_coupling_constant'] = pred['pred_fc'] + pred['pred_sd'] + pred['pred_pso'] + pred['pred_dso']
pred = pred[['id', 'scalar_coupling_constant']]
pred.to_csv(output_dir + '/t0_submission.csv', index=False)
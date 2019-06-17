import numpy as np
import pandas as pd

input_dir = '../work'

fc = pd.read_csv(input_dir + '/t0_predict_fc.csv')
sd = pd.read_csv(input_dir + '/t0_predict_sd.csv')
pso = pd.read_csv(input_dir + '/t0_predict_pso.csv')
dso = pd.read_csv(input_dir + '/t0_predict_dso.csv')

pred = fc.copy()
pred['sd'] = sd['pred_sd']
pred['pso'] = sd['pred_pso']
pred['dso'] = sd['pred_dso']

print(pred)
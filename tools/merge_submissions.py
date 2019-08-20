import numpy as np
import pandas as pd

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work/t4'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t4'

test = pd.read_csv(INPUT_DIR + '/test.csv')

base = pd.read_csv(WORK_DIR + '/t4c_scc_test_preungvar_orig.csv')

type_2jhc = pd.read_csv(WORK_DIR + '/t4c_scc_test.csv')

# X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb_oof['oof']

base.loc[test['type'] == '2JHC', 'scalar_coupling_constant'] = type_2jhc[test['type'] == '2JHC']['scalar_coupling_constant']

base.to_csv(f'{OUTPUT_DIR}/t4c_scc_merged_2jhc.csv', index=False)
import numpy as np
import pandas as pd
import os
import sys

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work'

# TYPE_WL = ['1JHN','2JHN','3JHN','2JHH','3JHH','1JHC','2JHC','3JHC']
TYPE_WL = ['1JHC','2JHC','3JHC', '1JHN','2JHN','3JHN','2JHH','3JHH']
# TYPE_WL = ['1JHC']

TARGET_WL = ['mulliken_charge']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 7,
    '1JHN': 5,
    # '2JHN': 7,
    # '3JHN': 7,
}

N_ESTIMATORS = {'_': 16000}

PARAMS = {
    '_': {
        'num_leaves': 128,
        'min_child_samples': 9,
        'objective': 'regression',
        'max_depth': 9,
        'learning_rate': 0.1,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": SEED,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'colsample_bytree': 1.0
    },
    '1JHN': {'subsample': 1, 'learning_rate': 0.05},
    '2JHN': {'subsample': 1, 'learning_rate': 0.05},
    '3JHN': {'subsample': 1, 'learning_rate': 0.05},
}
##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)

from edgar_playground.t3_lib_mull_v2 import train_model_regression
from edgar_playground.t3_lib_mull_v2 import t3_load_data
from edgar_playground.t3_lib_mull_v2 import t3_preprocess_data
from edgar_playground.t3_lib_mull_v2 import t3_create_features
from edgar_playground.t3_lib_mull_v2 import t3_prepare_columns
from edgar_playground.t3_lib_mull_v2 import t3_to_parquet
from edgar_playground.t3_lib_mull_v2 import t3_read_parquet

##### COPY__PASTE__LIB__END #####


train, test, structures, mulliken_charges = t3_load_data(INPUT_DIR)

train, test, structures = t3_preprocess_data(train, test, structures, mulliken_charges)

t3_create_features(train, test)

# t3_to_parquet(WORK_DIR, train, test, sub, structures, contributions, mulliken_charges)

# train, test, sub, structures, contributions, potential_energy, mulliken_charges = t3_read_parquet(WORK_DIR)

X, X_test, labels = t3_prepare_columns(train, test)

# X.to_csv(f'{OUTPUT_DIR}/t3_mull_v2_X.csv', index=False)
# X_test.to_csv(f'{OUTPUT_DIR}/t3_mull_v2_X_test.csv', index=False)

for type_name in TYPE_WL:
    for target in TARGET_WL:

        _PARAMS = {**PARAMS['_'], **PARAMS[type_name]} if type_name in PARAMS.keys() else PARAMS['_']
        _N_FOLD = N_FOLD[type_name] if type_name in N_FOLD.keys() else N_FOLD['_']
        _N_ESTIMATORS = N_ESTIMATORS[type_name] if type_name in N_ESTIMATORS.keys() else N_ESTIMATORS['_']

        y_target = train[target]
        t = labels['type'].transform([type_name])[0]

        folds = KFold(n_splits=_N_FOLD, shuffle=True, random_state=SEED)

        X_short = pd.DataFrame({
            'ind': list(X.index),
            'type': X['type'].values,
            'oof': [0] * len(X),
            'target': y_target.values})

        X_short_test = pd.DataFrame({
            'ind': list(X_test.index),
            'type': X_test['type'].values,
            'prediction': [0] * len(X_test)})

        X_t = X.loc[X['type'] == t]
        X_test_t = X_test.loc[X_test['type'] == t]
        y_t = X_short.loc[X_short['type'] == t, 'target']

        print("Training of type %s, component '%s', train size: %d" % (type_name, target, len(y_t)))

        result_dict_lgb_oof = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=_PARAMS, folds=folds,
                                                     model_type='lgb', eval_metric='group_mae', plot_feature_importance=False,
                                                     verbose=100, early_stopping_rounds=200, n_estimators=_N_ESTIMATORS)

        X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb_oof['oof']
        X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb_oof['prediction']

        train.loc[train['type'] == t, f'oof_{target}'] = X_short.loc[X_short['type'] == t, 'oof']
        test.loc[test['type'] == t, f'oof_{target}'] = X_short_test.loc[X_short_test['type'] == t, 'prediction']

train = train[['molecule_name', 'atom_index_0', 'oof_mulliken_charge']].rename(columns={'atom_index_0': 'atom_index'})
mean = train.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].mean().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_mean'})
median = train.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].median().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_median'})
std = train.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].std().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_std'})
pd.concat([mean, median, std], axis=1).to_csv(f'{OUTPUT_DIR}/t3_mull_v2_train.csv', index=True)

test = test[['molecule_name', 'atom_index_0', 'oof_mulliken_charge']].rename(columns={'atom_index_0': 'atom_index'})
mean = test.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].mean().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_mean'})
median = test.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].median().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_median'})
std = test.groupby(['molecule_name', 'atom_index'])[['oof_mulliken_charge']].std().rename(columns={'oof_mulliken_charge': 'oof_mulliken_charge_std'})
pd.concat([mean, median, std], axis=1).to_csv(f'{OUTPUT_DIR}/t3_mull_v2_test.csv', index=True)


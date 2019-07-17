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

TYPE_WL = ['1JHC','2JHC','3JHC','1JHN','2JHN','3JHN','2JHH','3JHH']
# TYPE_WL = ['1JHC','2JHC','3JHC']

TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 5,
    # '_': 7,
    # '1JHC': 7,
    # '1JHN': 5,
    # '2JHN': 7,
    # '3JHN': 7,
}

N_ESTIMATORS = {'_': 8000}

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

from edgar_playground.t3_lib_baseline import train_model_regression
from edgar_playground.t3_lib_baseline import t3_load_data
from edgar_playground.t3_lib_baseline import t3_preprocess_data
from edgar_playground.t3_lib_baseline import t3_create_features
from edgar_playground.t3_lib_baseline import t3_prepare_columns
from edgar_playground.t3_lib_baseline import t3_to_parquet
from edgar_playground.t3_lib_baseline import t3_read_parquet

##### COPY__PASTE__LIB__END #####


# train, test, sub, structures, contributions, mulliken_charges = t3_load_data(INPUT_DIR)
#
# train, test, structures = t3_preprocess_data(train, test, structures, contributions, mulliken_charges)
#
# t3_create_features(train, test)
#
# t3_to_parquet(WORK_DIR, train, test, sub, structures, contributions, mulliken_charges)

train, test, sub, structures, contributions, mulliken_charges = t3_read_parquet(WORK_DIR)

##### MULLIKEN #####

mulliken_charges = pd.read_csv(WORK_DIR + '/t3_mull_v2_train.csv')
train = pd.merge(train, mulliken_charges, how='left',
                 left_on=['molecule_name', 'atom_index_0'],
                 right_on=['molecule_name', 'atom_index'])
train.drop('atom_index', axis=1, inplace=True)
train.rename(inplace=True, columns={'oof_mulliken_charge_median': 'mulliken_charge_0'})

train = pd.merge(train, mulliken_charges, how='left',
                left_on=['molecule_name', 'atom_index_1'],
                right_on=['molecule_name', 'atom_index'])
train.drop('atom_index', axis=1, inplace=True)
train.rename(inplace=True, columns={'oof_mulliken_charge_median': 'mulliken_charge_1'})


mulliken_charges = pd.read_csv(WORK_DIR + '/t3_mull_v2_test.csv')
test = pd.merge(test, mulliken_charges, how='left',
                 left_on=['molecule_name', 'atom_index_0'],
                 right_on=['molecule_name', 'atom_index'])
test.drop('atom_index', axis=1, inplace=True)
test.rename(inplace=True, columns={'oof_mulliken_charge_median': 'mulliken_charge_0'})

test = pd.merge(test, mulliken_charges, how='left',
                left_on=['molecule_name', 'atom_index_1'],
                right_on=['molecule_name', 'atom_index'])
test.drop('atom_index', axis=1, inplace=True)
test.rename(inplace=True, columns={'oof_mulliken_charge_median': 'mulliken_charge_1'})

##### /MULLIKEN #####

##### CONTRIBUTIONS #####

train.drop(columns=['fc','sd','pso','dso'], inplace=True)

oof_contributions = pd.read_csv(WORK_DIR + '/t3_baseline_mull_train.csv')
train = pd.merge(train, oof_contributions, how='left',
                 left_on=['id'],
                 right_on=['id'])
train.rename(inplace=True, columns={
    'oof_fc': 'fc',
    'oof_sd': 'sd',
    'oof_pso': 'pso',
    'oof_dso': 'dso',
    })

oof_contributions = pd.read_csv(WORK_DIR + '/t3_baseline_mull_test.csv')
test = pd.merge(test, oof_contributions, how='left',
                 left_on=['id'],
                 right_on=['id'])
test.rename(inplace=True, columns={
    'oof_fc': 'fc',
    'oof_sd': 'sd',
    'oof_pso': 'pso',
    'oof_dso': 'dso',
    })

##### /CONTRIBUTIONS #####

X, X_test, labels = t3_prepare_columns(train, test,
                                          good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1', 'fc', 'sd',
                                                              'pso', 'dso'])

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

train[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t3_scc_train.csv', index=False)
test.rename(inplace=True, columns={'oof_scalar_coupling_constant': 'scalar_coupling_constant',})
test[['id'] + [f'{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t3_scc_test.csv', index=False)

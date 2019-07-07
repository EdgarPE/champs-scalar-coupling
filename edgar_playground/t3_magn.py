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
TYPE_WL = ['1JHC']

TARGET_WL = [
    'magnetic_st_0_XX',
    # 'magnetic_st_0_XY',
    # 'magnetic_st_0_XZ',
    # 'magnetic_st_0_YX',
    # 'magnetic_st_0_YY',
    # 'magnetic_st_0_YZ',
    # 'magnetic_st_0_ZX',
    # 'magnetic_st_0_ZY',
    # 'magnetic_st_0_ZZ',
    # 'magnetic_st_1_XX',
    # 'magnetic_st_1_XY',
    # 'magnetic_st_1_XZ',
    # 'magnetic_st_1_YX',
    # 'magnetic_st_1_YY',
    # 'magnetic_st_1_YZ',
    # 'magnetic_st_1_ZX',
    # 'magnetic_st_1_ZY',
    # 'magnetic_st_1_ZZ',
]

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 3,
    # '1JHN': 7,
    # '2JHN': 7,
    # '3JHN': 7,
}

N_ESTIMATORS = {'_': 10000}

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

from edgar_playground.t3_lib import train_model_regression
from edgar_playground.t3_lib import t3_load_data
from edgar_playground.t3_lib import t3_preprocess_data
from edgar_playground.t3_lib import t3_create_features
from edgar_playground.t3_lib import t3_prepare_columns
from edgar_playground.t3_lib import t3_to_parquet
from edgar_playground.t3_lib import t3_read_parquet

##### COPY__PASTE__LIB__END #####


train, test, sub, structures, contributions, potential_energy, mulliken_charges, dipole_moments, magnetic_st = t3_load_data(INPUT_DIR)

train, test = t3_preprocess_data(train, test, structures, contributions, potential_energy, mulliken_charges, dipole_moments, magnetic_st)

train['potential_energy'] = 0       # 1JHC-fc predict \w potential_energy:      no change
# train['mulliken_charge_0'] = 0      # 1JHC-fc predict \w mulliken_*             L1: 2.02 => 1.67
# train['mulliken_charge_1'] = 0
train['dipole_moment_X'] = 0        # 1JHC-fc predict \w dipole_*             no change, even worse?
train['dipole_moment_Y'] = 0
train['dipole_moment_Z'] = 0
# train['magnetic_st_0_XX'] = 0         # 1JHC-fc predict \w magnetic_st_*        L1: 2.02 => 1.76
# train['magnetic_st_0_XY'] = 0
# train['magnetic_st_0_XZ'] = 0
# train['magnetic_st_0_YX'] = 0
# train['magnetic_st_0_YY'] = 0
# train['magnetic_st_0_YZ'] = 0
# train['magnetic_st_0_ZX'] = 0
# train['magnetic_st_0_ZY'] = 0
# train['magnetic_st_0_ZZ'] = 0
# train['magnetic_st_1_XX'] = 0
# train['magnetic_st_1_XY'] = 0
# train['magnetic_st_1_XZ'] = 0
# train['magnetic_st_1_YX'] = 0
# train['magnetic_st_1_YY'] = 0
# train['magnetic_st_1_YZ'] = 0
# train['magnetic_st_1_ZX'] = 0
# train['magnetic_st_1_ZY'] = 0
# train['magnetic_st_1_ZZ'] = 0


test['potential_energy'] = 0
test['mulliken_charge_0'] = 0
test['mulliken_charge_1'] = 0
test['dipole_moment_X'] = 0
test['dipole_moment_Y'] = 0
test['dipole_moment_Z'] = 0
test['magnetic_st_0_XX'] = 0
test['magnetic_st_0_XY'] = 0
test['magnetic_st_0_XZ'] = 0
test['magnetic_st_0_YX'] = 0
test['magnetic_st_0_YY'] = 0
test['magnetic_st_0_YZ'] = 0
test['magnetic_st_0_ZX'] = 0
test['magnetic_st_0_ZY'] = 0
test['magnetic_st_0_ZZ'] = 0
test['magnetic_st_1_XX'] = 0
test['magnetic_st_1_XY'] = 0
test['magnetic_st_1_XZ'] = 0
test['magnetic_st_1_YX'] = 0
test['magnetic_st_1_YY'] = 0
test['magnetic_st_1_YZ'] = 0
test['magnetic_st_1_ZX'] = 0
test['magnetic_st_1_ZY'] = 0
test['magnetic_st_1_ZZ'] = 0

t3_create_features(train, test)

# t3_to_parquet(WORK_DIR, train, test, sub, structures, contributions, potential_energy, mulliken_charges)

# train, test, sub, structures, contributions, potential_energy, mulliken_charges = t3_read_parquet(WORK_DIR)

X, X_test, labels = t3_prepare_columns(train, test)

# X.to_csv(f'{OUTPUT_DIR}/t3_mull_X.csv', index=False)
# X_test.to_csv(f'{OUTPUT_DIR}/t3_mull_X_test.csv', index=False)

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

train[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t3_magnetic_st_train.csv', index=False)
test[['id'] + [f'oof_{c}' for c in TARGET_WL]].to_csv(f'{OUTPUT_DIR}/t3_magnetic_st_test.csv', index=False)

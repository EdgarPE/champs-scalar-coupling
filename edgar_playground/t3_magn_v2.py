import numpy as np
import pandas as pd
import os
import sys

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

##############################################################################
# Prob A: 32 GB ram nem elég, az oof becslésen elbukik (CV mean score után)
# Prob B: rosszul becsülhető, 1JHC - magnetic_st_XX | fold 3 | 4000 extimator:

# Training of type 1JHC, component 'magnetic_st_XX', train size: 1418832
# Fold 1 started at Wed Jul 17 22:43:18 2019
# Training until validation scores don't improve for 200 rounds.
# [100]	training's l1: 5.98399	valid_1's l1: 6.1196
# [200]	training's l1: 5.56248	valid_1's l1: 5.82359
# [300]	training's l1: 5.25096	valid_1's l1: 5.61769
# [400]	training's l1: 5.02444	valid_1's l1: 5.48802
# [500]	training's l1: 4.82159	valid_1's l1: 5.37115
# [600]	training's l1: 4.65535	valid_1's l1: 5.28391
# [700]	training's l1: 4.49874	valid_1's l1: 5.20334
# [800]	training's l1: 4.3624	valid_1's l1: 5.13964
# [900]	training's l1: 4.23344	valid_1's l1: 5.07755
# [1000]	training's l1: 4.11723	valid_1's l1: 5.02556
# [1100]	training's l1: 4.01237	valid_1's l1: 4.98112
# [1200]	training's l1: 3.91517	valid_1's l1: 4.94255
# [1300]	training's l1: 3.82228	valid_1's l1: 4.90363
# [1400]	training's l1: 3.734	valid_1's l1: 4.86785
# [1500]	training's l1: 3.65114	valid_1's l1: 4.83305
# [1600]	training's l1: 3.57313	valid_1's l1: 4.80334
# [1700]	training's l1: 3.49754	valid_1's l1: 4.77553
# [1800]	training's l1: 3.42172	valid_1's l1: 4.74568
# [1900]	training's l1: 3.35105	valid_1's l1: 4.71881
# [2000]	training's l1: 3.28606	valid_1's l1: 4.6941
# [2100]	training's l1: 3.22468	valid_1's l1: 4.67321
# [2200]	training's l1: 3.16194	valid_1's l1: 4.64978
# [2300]	training's l1: 3.10092	valid_1's l1: 4.62535
# [2400]	training's l1: 3.04262	valid_1's l1: 4.60412
# [2500]	training's l1: 2.98976	valid_1's l1: 4.58579
# [2600]	training's l1: 2.9372	valid_1's l1: 4.56703
# [2700]	training's l1: 2.88557	valid_1's l1: 4.54953
# [2800]	training's l1: 2.83654	valid_1's l1: 4.53296
# [2900]	training's l1: 2.78991	valid_1's l1: 4.51749
# [3000]	training's l1: 2.74486	valid_1's l1: 4.50286
# [3100]	training's l1: 2.69939	valid_1's l1: 4.48804
# [3200]	training's l1: 2.65651	valid_1's l1: 4.47458
# [3300]	training's l1: 2.61435	valid_1's l1: 4.45965
# [3400]	training's l1: 2.5726	valid_1's l1: 4.44511
# [3500]	training's l1: 2.53189	valid_1's l1: 4.43129
# [3600]	training's l1: 2.4917	valid_1's l1: 4.41791
# [3700]	training's l1: 2.45384	valid_1's l1: 4.40519
# [3800]	training's l1: 2.41617	valid_1's l1: 4.39276
# [3900]	training's l1: 2.37827	valid_1's l1: 4.37931
# [4000]	training's l1: 2.34306	valid_1's l1: 4.36856
# Did not meet early stopping. Best iteration is:
# [4000]	training's l1: 2.34306	valid_1's l1: 4.36856

##############################################################################


INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work'

# TYPE_WL = ['1JHC','2JHC','3JHC', '1JHN','2JHN','3JHN','2JHH','3JHH']
TYPE_WL = ['1JHC']

TARGET_WL = [
    'magnetic_st_XX',
    # 'magnetic_st_XY',
    # 'magnetic_st_XZ',

    # 'magnetic_st_YX',
    'magnetic_st_YY',
    # 'magnetic_st_YZ',

    # 'magnetic_st_ZX',
    # 'magnetic_st_ZY',
    'magnetic_st_ZZ',
]

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 3,
    '1JHN': 5,
    # '2JHN': 7,
    # '3JHN': 7,
}

N_ESTIMATORS = {'_': 4000}

PARAMS = {
    '_': {
        'num_leaves': 500,
        'min_child_samples': 79,
        'objective': 'regression',
        'max_depth': 12,
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

from edgar_playground.t3_lib_magn_v2 import train_model_regression
from edgar_playground.t3_lib_magn_v2 import t3_load_data
from edgar_playground.t3_lib_magn_v2 import t3_preprocess_data
from edgar_playground.t3_lib_magn_v2 import t3_create_features
from edgar_playground.t3_lib_magn_v2 import t3_prepare_columns
from edgar_playground.t3_lib_magn_v2 import t3_to_parquet
from edgar_playground.t3_lib_magn_v2 import t3_read_parquet

##### COPY__PASTE__LIB__END #####


train, test, structures, magnetic_st = t3_load_data(INPUT_DIR)

train, test, structures = t3_preprocess_data(train, test, structures, magnetic_st)

t3_create_features(train, test)

# t3_to_parquet(WORK_DIR, train, test, sub, structures, contributions, magnetic_st)

# train, test, sub, structures, contributions, potential_energy, magnetic_st = t3_read_parquet(WORK_DIR)

X, X_test, labels = t3_prepare_columns(train, test)

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

train = train[['molecule_name', 'atom_index_0', 'oof_magnetic_st_XX', 'oof_magnetic_st_YY', 'oof_magnetic_st_ZZ']].rename(columns={'atom_index_0': 'atom_index'})
magnetic_st_XX = train.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_XX']].median()
magnetic_st_YY = train.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_YY']].median()
magnetic_st_ZZ = train.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_ZZ']].median()
pd.concat([magnetic_st_XX, magnetic_st_YY, magnetic_st_ZZ], axis=1).to_csv(f'{OUTPUT_DIR}/t3_magn_v2_train.csv', index=True)

test = test[['molecule_name', 'atom_index_0', 'oof_magnetic_st_XX', 'oof_magnetic_st_YY', 'oof_magnetic_st_ZZ']].rename(columns={'atom_index_0': 'atom_index'})
magnetic_st_XX = test.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_XX']].median()
magnetic_st_YY = test.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_YY']].median()
magnetic_st_ZZ = test.groupby(['molecule_name', 'atom_index'])[['oof_magnetic_st_ZZ']].median()
pd.concat([magnetic_st_XX, magnetic_st_YY, magnetic_st_ZZ], axis=1).to_csv(f'{OUTPUT_DIR}/t3_magn_v2_test.csv', index=True)


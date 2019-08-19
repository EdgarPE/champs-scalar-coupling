import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import ShuffleSplit

##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t4_lib import *

##### COPY__PASTE__LIB__END #####

INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

# WORK_DIR= '.'
WORK_DIR = '../work/t4'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t4'

# TYPE_WL = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
TYPE_WL = ['3JHN', '2JHH', '3JHH', '1JHC', '2JHC', '3JHC']

# TARGET_WL = ['fc', 'sd', 'pso', 'dso']
TARGET_WL = ['fc']

SEED = 55
np.random.seed(SEED)

N_FOLD = {
    '_': 3,
}

N_ESTIMATORS = {
    '_': 300, # 8000-nek még van értelme
}


FIX_PARAMS = {
    '_': {
        # 'num_leaves': 128,
        # 'min_child_samples': 79,
        # 'objective': 'regression',
        # 'max_depth': 9,
        # 'learning_rate': 0.1,
        # "boosting_type": "gbdt",
        # "subsample_freq": 1,
        # "subsample": 0.9,
        # "bagging_seed": SEED,
        # "metric": 'mae',
        # "verbosity": -1,
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 0.3,
        # 'colsample_bytree': 1.0
    },
    # '1JHN': {'subsample': 1, 'learning_rate': 0.05},
    # '2JHN': {'subsample': 1, 'learning_rate': 0.05},
    # '3JHN': {'subsample': 1, 'learning_rate': 0.05},
    '1JHN': {'subsample': 1, },
    '2JHN': {'subsample': 1, },
    '3JHN': {'subsample': 1, },
    # '1JHC': {'min_child_samples': 120},
}

FIX_PARAMS = {
    '_': {
            "early_stopping_rounds": 200,
            "eval_metric" : 'mae',
            # 'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto',
          },
}

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

SEARCH_PARAMS = {
    '_': {
        # 'min_child_samples': sp_randint(25, 150),
        # 'num_leaves': [50, 100, 150, 200, 300, 500],
        # 'subsample': [0.2, 0.4, 0.6, 0.8, 0.9, 1],
        'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
        'reg_alpha': sp_uniform(loc=0.0, scale=0.5),
        'reg_lambda': sp_uniform(loc=0.0, scale=0.6),
    },
}

# train, test, structures, contributions = t4_load_data(INPUT_DIR)
#
# train, test = t4_criskiev_features(train, test, structures)
#
# structures = t4_merge_yukawa(INPUT_DIR, structures)
#
# structures = t4_crane_features(structures)
#
# train, test = t4_merge_structures(train, test, structures)
#
# t4_distance_feature(train, test)
#
# t4_artgor_features(train, test)

#
# Save to and/or load from parquet
#
# t4_to_parquet(WORK_DIR, train, test, structures, contributions)

train, test, structures, contributions = t4_read_parquet(WORK_DIR)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t4_load_data_mulliken_oof(WORK_DIR, train, test)

#
# Merge contributions fact data
#
train = t4_merge_contributions(train, contributions)

#
# Run HPO
#
X, X_test, labels = t4_prepare_columns(train, test, good_columns_extra=['mulliken_charge_0', 'mulliken_charge_1'])

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

for type_name in TYPE_WL:
    for target in TARGET_WL:
        _FIX_PARAMS = {**FIX_PARAMS['_'], **FIX_PARAMS[type_name]} if type_name in FIX_PARAMS.keys() else FIX_PARAMS['_']
        _SEARCH_PARAMS = {**SEARCH_PARAMS['_'], **SEARCH_PARAMS[type_name]} if type_name in SEARCH_PARAMS.keys() else SEARCH_PARAMS['_']
        _N_FOLD = N_FOLD[type_name] if type_name in N_FOLD.keys() else N_FOLD['_']
        _N_ESTIMATORS = N_ESTIMATORS[type_name] if type_name in N_ESTIMATORS.keys() else N_ESTIMATORS['_']

        print(_FIX_PARAMS)
        print(_SEARCH_PARAMS)

        y_target = train[target]
        t = labels['type'].transform([type_name])[0]

        # folds = KFold(n_splits=_N_FOLD, shuffle=True, random_state=SEED)

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

        columns = X.columns

        print(X_t.shape)

        (train_index, valid_index) = next(ShuffleSplit(n_splits=1, train_size=0.8).split(X_t, y_t))

        if type(X) == np.ndarray:
            X_train, X_valid = X_t[columns][train_index], X_t[columns][valid_index]
            y_train, y_valid = y_t[train_index], y_t[valid_index]
        else:
            X_train, X_valid = X_t[columns].iloc[train_index], X_t[columns].iloc[valid_index]
            y_train, y_valid = y_t.iloc[train_index], y_t.iloc[valid_index]

        # eval_set = [(X_train, y_train), (X_valid, y_valid)]
        # _FIX_PARAMS['eval_set'] = (X_valid, y_valid)
        # _FIX_PARAMS['eval_set'] = [(X_train, y_train), (X_valid, y_valid)]
        _FIX_PARAMS['eval_set'] = [(X_valid, y_valid), (X_train, y_train)]

        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        print("Training of type %s, component '%s', train size: %d" % (type_name, target, len(y_t)))

        # result_dict_lgb_oof = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=_PARAMS, folds=folds,
        #                                              model_type='lgb', eval_metric='group_mae',
        #                                              plot_feature_importance=True,
        #                                              verbose=100, early_stopping_rounds=200,
        #                                              n_estimators=_N_ESTIMATORS)

        #This parameter defines the number of HP points to be tested
        n_HP_points_to_test = 30

        #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
        clf = lgb.LGBMRegressor(random_state=SEED, silent=True, metric='mae', n_jobs=-1, n_estimators=_N_ESTIMATORS,
                                max_depth=12, objective='regression', num_leaves=128, learning_rate=0.1, boosting_type='gbdt',
                                subsample_freq=1, subsample=1, bagging_seed=SEED, verbosity=-1,
                                min_child_samples=22) # colsample_bytree=1.0,  reg_alpha=0.1,  reg_lambda=0.3,

        gs = RandomizedSearchCV(
            estimator=clf, param_distributions=_SEARCH_PARAMS,
            n_iter=n_HP_points_to_test,
            scoring='neg_mean_absolute_error',
            cv=_N_FOLD,
            refit=True,
            random_state=SEED,
            verbose=10000)

        gs.fit(X_train, y_train, **_FIX_PARAMS)
        _FIX_PARAMS['eval_set'] = None
        print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
        # print('Results:')
        # print(gs.cv_results_)

        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in sorted(zip(means, stds, gs.cv_results_['params']), key=lambda x: list(x[2].values())[0],
                                        reverse=False):
            print("%0.6f (+/-%0.06f) for %r"
                  % (mean, std * 2, params))

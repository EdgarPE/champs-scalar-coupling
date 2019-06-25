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
TYPE_WL = ['2JHH','3JHH']

TARGET_WL = ['fc']

N_FOLD = 3

N_ESTIMATORS = 3000

SEED = 55
np.random.seed(SEED)

# params = {'num_leaves': 128,
#           'min_child_samples': 79,
#           'objective': 'regression',
#           'max_depth': 9,
#           'learning_rate': 0.5,
#           "boosting_type": "gbdt",
#           "subsample_freq": 1,
#           "subsample": 0.9,
#           "bagging_seed": SEED,
#           "metric": 'mae',
#           "verbosity": -1,
#           'reg_alpha': 0.1,
#           'reg_lambda': 0.3,
#           'colsample_bytree': 1.0
#           }

params = {'num_leaves': 255,
          'min_child_samples': 39,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.3,
          "boosting_type": "gbdt",
          "subsample_freq": 0,
          "subsample": 0.9,
          "bagging_seed": SEED,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0
          }
##### COPY__PASTE__LIB__BEGIN #####

basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)

from edgar_playground.t1_lib import train_model_regression
from edgar_playground.t1_lib import t1_load_data
from edgar_playground.t1_lib import t1_preprocess_data
from edgar_playground.t1_lib import t1_create_features
from edgar_playground.t1_lib import t1_prepare_columns
from edgar_playground.t1_lib import t1_to_parquet
from edgar_playground.t1_lib import t1_read_parquet

##### COPY__PASTE__LIB__END #####


# train, test, sub, structures, contributions = t1_load_data(INPUT_DIR)
#
# train, test = t1_preprocess_data(train, test, structures, contributions)
#
# t1_create_features(train, test)
#
# t1_to_parquet(WORK_DIR, train, test, sub, structures, contributions)

train, test, sub, structures, contributions = t1_read_parquet(WORK_DIR)

X, X_test, y, labels = t1_prepare_columns(train, test)

for type_name in TYPE_WL:
    for target in TARGET_WL:

        y_target = train[target]
        t = labels['type'].transform([type_name])[0]

        folds = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

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

        print('Training of type %s, train size: %d' % (type_name, len(y_t)))

        result_dict_lgb_oof = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds,
                                                  model_type='lgb', eval_metric='group_mae', plot_feature_importance=False,
                                                  verbose=500, early_stopping_rounds=200, n_estimators=N_ESTIMATORS)
        X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb_oof['oof']
        X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb_oof['prediction']

        train[f'oof_{target}'] = X_short['oof']
        test[f'oof_{target}'] = X_short_test['prediction']

        train[['id', f'oof_{target}']].to_csv(f'{OUTPUT_DIR}/t1_{type_name}_oof_{target}_train.csv', index=False)
        test[['id', f'oof_{target}']].to_csv(f'{OUTPUT_DIR}/t1_{type_name}_oof_{target}_test.csv', index=False)
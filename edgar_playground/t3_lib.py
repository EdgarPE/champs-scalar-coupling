import numpy as np
import pandas as pd
import gc
import resource

##### COPY__PASTE__LIB__BEGIN #####

import time
from numba import jit
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

def map_atom_info(df, atom_idx, structures):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:3] == 'int' or str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None,
                           plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                      **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            # plt.figure(figsize=(16, 12));
            # sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            # plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance

    return result_dict


def create_features(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']
    # df = reduce_mem_usage(df)

    # return df


def t3_load_data(input_dir):
    train = pd.read_csv(input_dir + '/train.csv')
    test = pd.read_csv(input_dir + '/test.csv')
    sub = pd.read_csv(input_dir + '/sample_submission.csv')
    structures = pd.read_csv(input_dir + '/structures.csv')
    contributions = pd.read_csv(input_dir + '/scalar_coupling_contributions.csv')

    # dipole_moments = pd.read_csv(input_dir + '/dipole_moments.csv')
    # magnetic_shielding_tensors = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')
    # mulliken_charges = pd.read_csv(input_dir + '/mulliken_charges.csv')
    # potential_energy = pd.read_csv(input_dir + '/potential_energy.csv')

    print('Train dataset shape is now rows: {} cols:{}'.format(train.shape[0], train.shape[1]))
    print('Test dataset shape is now rows: {} cols:{}'.format(test.shape[0], test.shape[1]))
    print('Sub dataset shape is now rows: {} cols:{}'.format(sub.shape[0], sub.shape[1]))
    print('Structures dataset shape is now rows: {} cols:{}'.format(structures.shape[0], structures.shape[1]))
    print('Scalar_coupling_contributions dataset shape is now rows: {} cols:{}'.format(contributions.shape[0],
                                                                                       contributions.shape[1]))

    return train, test, sub, structures, contributions


def t3_preprocess_data(train, test, structures, contributions):

    train = pd.merge(train, contributions, how='left',
                    left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                    right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

    # train = pd.merge(train, potential_energy, how='left',
    #                 left_on=['molecule_name'],
    #                 right_on=['molecule_name'])
    #
    # train = pd.merge(train, mulliken_charges, how='left',
    #                 left_on=['molecule_name', 'atom_index_0'],
    #                 right_on=['molecule_name', 'atom_index'])
    #
    # train = train.rename(columns={'mulliken_charge': 'mulliken_charge_0'})
    #
    # train = pd.merge(train, mulliken_charges, how='left',
    #                 left_on=['molecule_name', 'atom_index_1'],
    #                 right_on=['molecule_name', 'atom_index'])
    #
    # train = train.rename(columns={'mulliken_charge': 'mulliken_charge_1'})

    train = map_atom_info(train, 0, structures)
    train = map_atom_info(train, 1, structures)

    test = map_atom_info(test, 0, structures)
    test = map_atom_info(test, 1, structures)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    return train, test


def t3_create_features(train, test):
    create_features(train)
    create_features(test)


def t3_to_parquet(work_dir, train, test, sub, structures, contributions):
    train.to_parquet(f'{work_dir}/t3_train.parquet')
    test.to_parquet(f'{work_dir}/t3_test.parquet')
    sub.to_parquet(f'{work_dir}/t3_sub.parquet')
    structures.to_parquet(f'{work_dir}/t3_structures.parquet')
    contributions.to_parquet(f'{work_dir}/t3_contributions.parquet')


def t3_read_parquet(work_dir):
    train = pd.read_parquet(f'{work_dir}/t3_train.parquet')
    test = pd.read_parquet(f'{work_dir}/t3_test.parquet')
    sub = pd.read_parquet(f'{work_dir}/t3_sub.parquet')
    structures = pd.read_parquet(f'{work_dir}/t3_structures.parquet')
    contributions = pd.read_parquet(f'{work_dir}/t3_contributions.parquet')

    return train, test, sub, structures, contributions


def t3_prepare_columns(train, test):

    good_columns = [
        'molecule_atom_index_0_dist_min',
        'molecule_atom_index_0_dist_max',
        'molecule_atom_index_1_dist_min',
        'molecule_atom_index_0_dist_mean',
        'molecule_atom_index_0_dist_std',
        'dist',
        'molecule_atom_index_1_dist_std',
        'molecule_atom_index_1_dist_max',
        'molecule_atom_index_1_dist_mean',
        'molecule_atom_index_0_dist_max_diff',
        'molecule_atom_index_0_dist_max_div',
        'molecule_atom_index_0_dist_std_diff',
        'molecule_atom_index_0_dist_std_div',
        'atom_0_couples_count',
        'molecule_atom_index_0_dist_min_div',
        'molecule_atom_index_1_dist_std_diff',
        'molecule_atom_index_0_dist_mean_div',
        'atom_1_couples_count',
        'molecule_atom_index_0_dist_mean_diff',
        'molecule_couples',
        'atom_index_1',
        'molecule_dist_mean',
        'molecule_atom_index_1_dist_max_diff',
        'molecule_atom_index_0_y_1_std',
        'molecule_atom_index_1_dist_mean_diff',
        'molecule_atom_index_1_dist_std_div',
        'molecule_atom_index_1_dist_mean_div',
        'molecule_atom_index_1_dist_min_diff',
        'molecule_atom_index_1_dist_min_div',
        'molecule_atom_index_1_dist_max_div',
        'molecule_atom_index_0_z_1_std',
        'y_0',
        'molecule_type_dist_std_diff',
        'molecule_atom_1_dist_min_diff',
        'molecule_atom_index_0_x_1_std',
        'molecule_dist_min',
        'molecule_atom_index_0_dist_min_diff',
        'molecule_atom_index_0_y_1_mean_diff',
        'molecule_type_dist_min',
        'molecule_atom_1_dist_min_div',
        'atom_index_0',
        'molecule_dist_max',
        'molecule_atom_1_dist_std_diff',
        'molecule_type_dist_max',
        'molecule_atom_index_0_y_1_max_diff',
        'molecule_type_0_dist_std_diff',
        'molecule_type_dist_mean_diff',
        'molecule_atom_1_dist_mean',
        'molecule_atom_index_0_y_1_mean_div',
        'molecule_type_dist_mean_div',
        'type',
    ]

    labels = {}
    for f in ['atom_1', 'type_0', 'type']:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

            labels[f] = lbl

    X = train[good_columns].copy()
    y = train['scalar_coupling_constant']
    X_test = test[good_columns].copy()

    return X, X_test, y, labels

##### COPY__PASTE__LIB__END #####

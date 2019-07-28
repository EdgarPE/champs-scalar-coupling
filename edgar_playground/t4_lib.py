import numpy as np
import pandas as pd

##### COPY__PASTE__LIB__BEGIN #####

import time
from numba import jit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

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
    print(list(df.columns))
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    # Hiányzó, de nem lesz jobb
    # df[f'molecule_atom_index_0_x_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('mean')
    # df[f'molecule_atom_index_0_x_1_mean_diff'] = df[f'molecule_atom_index_0_x_1_mean'] - df['x_1']
    # df[f'molecule_atom_index_0_x_1_mean_div'] = df[f'molecule_atom_index_0_x_1_mean'] / df['x_1']
    # df[f'molecule_atom_index_0_x_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('max')
    # df[f'molecule_atom_index_0_x_1_max_diff'] = df[f'molecule_atom_index_0_x_1_max'] - df['x_1']

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    # Hiányzó, de nem lesz jobb
    # df[f'molecule_atom_index_0_z_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('mean')
    # df[f'molecule_atom_index_0_z_1_mean_diff'] = df[f'molecule_atom_index_0_z_1_mean'] - df['z_1']
    # df[f'molecule_atom_index_0_z_1_mean_div'] = df[f'molecule_atom_index_0_z_1_mean'] / df['z_1']
    # df[f'molecule_atom_index_0_z_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('max')
    # df[f'molecule_atom_index_0_z_1_max_diff'] = df[f'molecule_atom_index_0_z_1_max'] - df['z_1']

    # Hiányzó, de nem lesz jobb
    # df[f'molecule_atom_index_1_x_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('mean')
    # df[f'molecule_atom_index_1_x_0_mean_diff'] = df[f'molecule_atom_index_1_x_0_mean'] - df['x_0']
    # df[f'molecule_atom_index_1_x_0_mean_div'] = df[f'molecule_atom_index_1_x_0_mean'] / df['x_0']
    # df[f'molecule_atom_index_1_x_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['x_0'].transform('max')
    # df[f'molecule_atom_index_1_x_0_max_diff'] = df[f'molecule_atom_index_1_x_0_max'] - df['x_0']
    #
    # Hiányzó, de nem lesz jobb
    # df[f'molecule_atom_index_1_y_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('mean')
    # df[f'molecule_atom_index_1_y_0_mean_diff'] = df[f'molecule_atom_index_1_y_0_mean'] - df['y_0']
    # df[f'molecule_atom_index_1_y_0_mean_div'] = df[f'molecule_atom_index_1_y_0_mean'] / df['y_0']
    # df[f'molecule_atom_index_1_y_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['y_0'].transform('max')
    # df[f'molecule_atom_index_1_y_0_max_diff'] = df[f'molecule_atom_index_1_y_0_max'] - df['y_0']
    #
    # Hiányzó, de nem lesz jobb
    # df[f'molecule_atom_index_1_z_0_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('mean')
    # df[f'molecule_atom_index_1_z_0_mean_diff'] = df[f'molecule_atom_index_1_z_0_mean'] - df['z_0']
    # df[f'molecule_atom_index_1_z_0_mean_div'] = df[f'molecule_atom_index_1_z_0_mean'] / df['z_0']
    # df[f'molecule_atom_index_1_z_0_max'] = df.groupby(['molecule_name', 'atom_index_1'])['z_0'].transform('max')
    # df[f'molecule_atom_index_1_z_0_max_diff'] = df[f'molecule_atom_index_1_z_0_max'] - df['z_0']

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


def t4_load_data(input_dir):
    train = pd.read_csv(input_dir + '/train.csv')
    test = pd.read_csv(input_dir + '/test.csv')
    structures = pd.read_csv(input_dir + '/structures.csv')
    contributions = pd.read_csv(input_dir + '/scalar_coupling_contributions.csv')

    print('Train dataset shape is now rows: {} cols:{}'.format(train.shape[0], train.shape[1]))
    print('Test dataset shape is now rows: {} cols:{}'.format(test.shape[0], test.shape[1]))
    print('Structures dataset shape is now rows: {} cols:{}'.format(structures.shape[0], structures.shape[1]))
    print('Scalar_coupling_contributions dataset shape is now rows: {} cols:{}'.format(contributions.shape[0],
                                                                                       contributions.shape[1]))

    return train, test, structures, contributions


# https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
def t4_merge_yukawa(input_dir, structures):
    yukawa = pd.read_csv(input_dir + '/yukawa/structures_yukawa.csv')
    structures = pd.concat([structures, yukawa], axis=1)
    del yukawa
    print('Merge Yukawa. Structures dataset shape is now rows: {} cols:{}'.format(structures.shape[0],
                                                                                  structures.shape[1]))

    return structures


def t4_load_data_mulliken(input_dir):
    # dipole_moments = pd.read_csv(input_dir + '/dipole_moments.csv')
    # magnetic_st = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')
    mulliken_charges = pd.read_csv(input_dir + '/mulliken_charges.csv')
    # potential_energy = pd.read_csv(input_dir + '/potential_energy.csv')

    return mulliken_charges


def t4_load_data_mulliken_oof(work_dir, train, test):

    mulliken_charges = pd.read_csv(work_dir + '/t4a_mulliken_train.csv')
    train = pd.merge(train, mulliken_charges, how='left',
                     left_on=['molecule_name', 'atom_index_0'],
                     right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)
    train.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_0'})

    train = pd.merge(train, mulliken_charges, how='left',
                     left_on=['molecule_name', 'atom_index_1'],
                     right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)
    train.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_1'})

    mulliken_charges = pd.read_csv(work_dir + '/t4a_mulliken_test.csv')
    test = pd.merge(test, mulliken_charges, how='left',
                    left_on=['molecule_name', 'atom_index_0'],
                    right_on=['molecule_name', 'atom_index'])
    test.drop('atom_index', axis=1, inplace=True)
    test.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_0'})

    test = pd.merge(test, mulliken_charges, how='left',
                    left_on=['molecule_name', 'atom_index_1'],
                    right_on=['molecule_name', 'atom_index'])
    test.drop('atom_index', axis=1, inplace=True)
    test.rename(inplace=True, columns={'oof_mulliken_charge': 'mulliken_charge_1'})

    return train, test


def t4_load_data_contributions_oof(work_dir, train, test):

    train.drop(columns=['fc', 'sd', 'pso', 'dso'], inplace=True)

    oof_contributions = pd.read_csv(work_dir + '/t4b_contributions_train.csv')
    train = pd.merge(train, oof_contributions, how='left',
                     left_on=['id'],
                     right_on=['id'])
    train.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })

    oof_contributions = pd.read_csv(work_dir + '/t4b_contributions_test.csv')
    test = pd.merge(test, oof_contributions, how='left',
                    left_on=['id'],
                    right_on=['id'])
    test.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })

    return train, test


def t4_preprocess_add_flip_data(df):
    dfc = df.copy()
    dfc[['atom_index_0', 'atom_index_1']] = dfc[['atom_index_1', 'atom_index_0']]

    return pd.concat([df, dfc])


def t4_preprocess_data(train, test, structures, contributions, add_flip=False):

    if add_flip:
        train = t4_preprocess_add_flip_data(train)
        test = t4_preprocess_add_flip_data(test)

    train = pd.merge(train, contributions, how='left',
                    left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                    right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

    # electronegativity and atomic_radius
    # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning

    # from tqdm import tqdm_notebook as tqdm
    atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}  # Without fudge factor

    fudge_factor = 0.05
    atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
    # print(atomic_radius)

    electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

    # structures = pd.read_csv(structures, dtype={'atom_index':np.int8})

    atoms = structures['atom'].values
    atoms_en = [electronegativity[x] for x in atoms]
    atoms_rad = [atomic_radius[x] for x in atoms]

    structures['EN'] = atoms_en
    structures['rad'] = atoms_rad

    # print(structures.head())

    i_atom = structures['atom_index'].values
    p = structures[['x', 'y', 'z']].values
    p_compare = p
    m = structures['molecule_name'].values
    m_compare = m
    r = structures['rad'].values
    r_compare = r

    source_row = np.arange(len(structures))
    max_atoms = 28

    bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
    bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

    print('Calculating bonds')

    for i in range(max_atoms - 1):
        p_compare = np.roll(p_compare, -1, axis=0)
        m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)

        mask = np.where(m == m_compare, 1, 0)  # Are we still comparing atoms in the same molecule?
        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare

        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        target_row = source_row + i + 1  # Note: Will be out of bounds of bonds array for some values of i
        target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(structures),
                              target_row)  # If invalid target, write to dummy row

        source_atom = i_atom
        target_atom = i_atom + i + 1  # Note: Will be out of bounds of bonds array for some values of i
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0), max_atoms,
                               target_atom)  # If invalid target, write to dummy col

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

    print('Counting and condensing bonds')

    bonds_numeric = [[i for i, x in enumerate(row) if x] for row in bonds]
    bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]] for j, row in
                    enumerate(bond_dists)]
    bond_lengths_mean = [np.mean(x) for x in bond_lengths]
    bond_lengths_median = [np.median(x) for x in bond_lengths]
    bond_lengths_std = [np.std(x) for x in bond_lengths]
    n_bonds = [len(x) for x in bonds_numeric]

    # bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
    # bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

    bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean,
                 'bond_lengths_std': bond_lengths_std, 'bond_lengths_median': bond_lengths_median}
    bond_df = pd.DataFrame(bond_data)
    structures = structures.join(bond_df)
    # print(structures.head(20))


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
    train['dist'] = 1 / (train['dist'] ** 3) # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    test['dist'] = 1 / (test['dist'] ** 3)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    return train, test, structures


def t4_preprocess_data_mulliken(train, mulliken_charges):

    # train = pd.merge(train, potential_energy, how='left',
    #                 left_on=['molecule_name'],
    #                 right_on=['molecule_name'])

    train = pd.merge(train, mulliken_charges, how='left',
                    left_on=['molecule_name', 'atom_index_0'],
                    right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)
    # train.rename(inplace=True, columns={'mulliken_charge': 'mulliken_charge_0'})

    # train = pd.merge(train, mulliken_charges, how='left',
    #                 left_on=['molecule_name', 'atom_index_1'],
    #                 right_on=['molecule_name', 'atom_index'])
    # train.drop('atom_index', axis=1, inplace=True)
    # train.rename(inplace=True, columns={'mulliken_charge': 'mulliken_charge_1'})

    return train


def t4_create_features(train, test):
    create_features(train)
    create_features(test)


def t4_to_parquet(work_dir, train, test, structures, contributions, mulliken_charges):
    train.to_parquet(f'{work_dir}/t4_train.parquet')
    test.to_parquet(f'{work_dir}/t4_test.parquet')
    structures.to_parquet(f'{work_dir}/t4_structures.parquet')
    contributions.to_parquet(f'{work_dir}/t4_contributions.parquet')
    mulliken_charges.to_parquet(f'{work_dir}/t4_mulliken_charges.parquet')


def t4_read_parquet(work_dir):
    train = pd.read_parquet(f'{work_dir}/t4_train.parquet')
    test = pd.read_parquet(f'{work_dir}/t4_test.parquet')
    structures = pd.read_parquet(f'{work_dir}/t4_structures.parquet')
    contributions = pd.read_parquet(f'{work_dir}/t4_contributions.parquet')
    mulliken_charges = pd.read_parquet(f'{work_dir}/t4_mulliken_charges.parquet')

    return train, test, structures, contributions, mulliken_charges


def t4_prepare_columns(train, test, good_columns_extra=None):

    good_columns = [
        'bond_lengths_mean_y',
        'bond_lengths_median_y',
        'bond_lengths_std_y',
        'bond_lengths_mean_x',

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

        # Yukawa, todo: rename '_x' and '_y' to something meeningful
        'dist_C_0_x', 'dist_C_1_x', 'dist_C_2_x', 'dist_C_3_x', 'dist_C_4_x', 'dist_F_0_x', 'dist_F_1_x', 'dist_F_2_x',
        'dist_F_3_x', 'dist_F_4_x', 'dist_H_0_x', 'dist_H_1_x', 'dist_H_2_x', 'dist_H_3_x', 'dist_H_4_x', 'dist_N_0_x',
        'dist_N_1_x', 'dist_N_2_x', 'dist_N_3_x', 'dist_N_4_x', 'dist_O_0_x', 'dist_O_1_x', 'dist_O_2_x', 'dist_O_3_x',
        'dist_O_4_x',

        'dist_C_0_y', 'dist_C_1_y', 'dist_C_2_y', 'dist_C_3_y', 'dist_C_4_y', 'dist_F_0_y', 'dist_F_1_y', 'dist_F_2_y',
        'dist_F_3_y', 'dist_F_4_y', 'dist_H_0_y', 'dist_H_1_y', 'dist_H_2_y', 'dist_H_3_y', 'dist_H_4_y', 'dist_N_0_y',
        'dist_N_1_y', 'dist_N_2_y', 'dist_N_3_y', 'dist_N_4_y', 'dist_O_0_y', 'dist_O_1_y', 'dist_O_2_y', 'dist_O_3_y',
        'dist_O_4_y',
    ]

    good_columns += (good_columns_extra if good_columns_extra is not None else [])

    labels = {}
    for f in ['atom_1', 'type_0', 'type']:
        if f in good_columns:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

            labels[f] = lbl

    X = train[good_columns].copy()
    X_test = test[good_columns].copy()

    return X, X_test, labels


def t4_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels):
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
                                                         model_type='lgb', eval_metric='group_mae',
                                                         plot_feature_importance=False,
                                                         verbose=100, early_stopping_rounds=200,
                                                         n_estimators=_N_ESTIMATORS)

            X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb_oof['oof']
            X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb_oof['prediction']

            train.loc[train['type'] == t, f'oof_{target}'] = X_short.loc[X_short['type'] == t, 'oof']
            test.loc[test['type'] == t, f'oof_{target}'] = X_short_test.loc[X_short_test['type'] == t, 'prediction']

##### COPY__PASTE__LIB__END #####

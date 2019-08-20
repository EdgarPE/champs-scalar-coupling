import numpy as np
import pandas as pd
import sys
import psutil
import os

##### COPY__PASTE__LIB__BEGIN #####

import time
from numba import jit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def disp_mem_usage():
    # print(psutil.virtual_memory())  # physical memory usage
    print('[Memory] % 7.3f GB' % (psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30))


def map_atom_info(df, atom_idx, structures):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'], suffixes=["_a0", "_a1"])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
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
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

def mean_log_mae(y_true, y_pred):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    return np.log((y_true - y_pred).abs().mean())


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
    :params: verbose - parameters for gradient boosting models
    :params: early_stopping_rounds - parameters for gradient boosting models
    :params: n_estimators - parameters for gradient boosting models

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
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds
                      )
            # callbacks=[lgb.reset_parameter(learning_rate=t5_learning_rate_010_decay_power_0995)]
            # categorical_feature=['qcut_subtype_1','qcut_subtype_2']

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

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)')
            plt.savefig(fname = sys.argv[0] + '.importance.png')
            plt.close()
            # plt.close('all')

            # plt.show()

            result_dict['feature_importance'] = feature_importance

    return result_dict


def t5_load_data(input_dir):
    train = pd.read_csv(input_dir + '/train.csv')
    test = pd.read_csv(input_dir + '/test.csv')
    structures = pd.read_csv(input_dir + '/structures.csv')
    contributions = pd.read_csv(input_dir + '/scalar_coupling_contributions.csv')

    int32 = ['id']
    int8 = ['atom_index_0', 'atom_index_1']
    float32 = ['scalar_coupling_constant']

    train[int32] = train[int32].astype('int32')
    train[int8] = train[int8].astype('int8')

    train[float32] = train[float32].astype('float32')
    test[int32] = test[int32].astype('int32')
    test[int8] = test[int8].astype('int8')

    int8 = ['atom_index']
    float32 = ['x', 'y', 'z']

    structures[int8] = structures[int8].astype('int8')
    structures[float32] = structures[float32].astype('float32')

    int8 = ['atom_index_0', 'atom_index_1']
    float32 = ['fc', 'sd', 'pso', 'dso']

    contributions[int8] = contributions[int8].astype('int8')
    contributions[float32] = contributions[float32].astype('float32')

    return train, test, structures, contributions


def t5_load_feature_criskiev(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/criskiev/criskiev_train.parquet')
    test = pd.read_parquet(feature_dir + '/criskiev/criskiev_test.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t5_load_feature_crane(feature_dir, structures):
    crane = pd.read_parquet(feature_dir + '/crane/crane_structures.parquet')

    return structures.join(crane)


def t5_load_feature_artgor(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/artgor/artgor_train.parquet')
    test = pd.read_parquet(feature_dir + '/artgor/artgor_test.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


# https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction
def t5_merge_yukawa(input_dir, structures):
    yukawa = pd.read_csv(input_dir + '/yukawa/structures_yukawa.csv').astype('float32')
    structures = pd.concat([structures, yukawa], axis=1)

    return structures


def t5_load_feature_edgar(feature_dir, train_, test_):
    train = pd.read_parquet(feature_dir + '/edgar/edgar_train.parquet')
    test = pd.read_parquet(feature_dir + '/edgar/edgar_test.parquet')

    return pd.concat([train_, train], axis=1), pd.concat([test_, test], axis=1)


def t5_load_data_mulliken(input_dir):
    # dipole_moments = pd.read_csv(input_dir + '/dipole_moments.csv')
    # magnetic_st = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')
    mulliken_charges = pd.read_csv(input_dir + '/mulliken_charges.csv')
    # potential_energy = pd.read_csv(input_dir + '/potential_energy.csv')

    int8 = ['atom_index']
    float32 = ['mulliken_charge']

    mulliken_charges[int8] = mulliken_charges[int8].astype('int8')
    mulliken_charges[float32] = mulliken_charges[float32].astype('float32')

    return mulliken_charges


def t5_load_data_magnetic_st(input_dir):
    magnetic_st = pd.read_csv(input_dir + '/magnetic_shielding_tensors.csv')

    drop = ['XY', 'XZ', 'YX', 'YZ', 'ZX', 'ZY']
    magnetic_st.drop(columns=drop, inplace=True)

    int8 = ['atom_index']
    magnetic_st[int8] = magnetic_st[int8].astype('int8')

    float32 = ['XX', 'YY', 'ZZ']
    magnetic_st[float32] = magnetic_st[float32].astype('float32')

    return magnetic_st


def t5_load_data_mulliken_oof(work_dir, train, test):
    mulliken_charges = pd.read_csv(work_dir + '/t5a_mulliken_train.csv')
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

    mulliken_charges = pd.read_csv(work_dir + '/t5a_mulliken_test.csv')
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


def t5_load_data_magnetic_st_oof(work_dir, train, test):
    # todo
    pass

def t5_merge_contributions(train, contributions):
    train = pd.merge(train, contributions, how='left',
                     left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                     right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

    # train['contrib_sum'] = train['fc'] + train['sd'] + train['pso'] + train['dso']

    return train


def t5_load_data_contributions_oof(work_dir, train, test):
    oof_contributions = pd.read_csv(work_dir + '/t5b_contributions_train.csv')
    train = pd.merge(train, oof_contributions, how='left',
                     left_on=['id'],
                     right_on=['id'])
    train.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })
    train['contrib_sum'] = train['fc'] + train['sd'] + train['pso'] + train['dso']

    oof_contributions = pd.read_csv(work_dir + '/t5b_contributions_test.csv')
    test = pd.merge(test, oof_contributions, how='left',
                    left_on=['id'],
                    right_on=['id'])
    test.rename(inplace=True, columns={
        'oof_fc': 'fc',
        'oof_sd': 'sd',
        'oof_pso': 'pso',
        'oof_dso': 'dso',
    })
    test['contrib_sum'] = test['fc'] + test['sd'] + test['pso'] + test['dso']

    return train, test


def t5_preprocess_add_flip_data(df):
    dfc = df.copy()
    dfc[['atom_index_0', 'atom_index_1']] = dfc[['atom_index_1', 'atom_index_0']]

    return pd.concat([df, dfc])


def t5_merge_structures(train, test, structures, add_flip=False):
    if add_flip:
        train = t5_preprocess_add_flip_data(train)
        test = t5_preprocess_add_flip_data(test)

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
    train['dist'] = 1 / (train['dist'] ** 3)  # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    test['dist'] = 1 / (test['dist'] ** 3)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    return train, test


def t5_distance_feature(train, test):
    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist_lin'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist_lin'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist'] = 1 / (train['dist_lin'] ** 3)  # https://www.kaggle.com/vaishvik25/1-r-3-hyperpar-tuning
    test['dist'] = 1 / (test['dist_lin'] ** 3)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    train['subtype'] = (train['type'].eq('1JHC') & train['dist_lin'].gt(1.065)).astype('int8')
    test['subtype'] = (test['type'].eq('1JHC') & test['dist_lin'] > 1.065).astype('int8')


def t5_preprocess_data_mulliken(train, mulliken_charges):
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


def t5_preprocess_data_magnetic_st(train, magnetic_st):
    train = pd.merge(train, magnetic_st, how='left',
                     left_on=['molecule_name', 'atom_index_0'],
                     right_on=['molecule_name', 'atom_index'])
    train.drop('atom_index', axis=1, inplace=True)

    return train


def t5_prepare_columns(train, test, good_columns_extra=None):
    good_columns = [
        # 'bond_lengths_mean_y',
        # 'bond_lengths_median_y',
        # 'bond_lengths_std_y',
        # 'bond_lengths_mean_x',

        'molecule_atom_index_0_dist_min',
        'molecule_atom_index_0_dist_max',
        'molecule_atom_index_1_dist_min',
        'molecule_atom_index_0_dist_mean',
        'molecule_atom_index_0_dist_std',
        'dist',
        'dist_lin',
        'subtype',
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

        # Crane
        'dist_C_0_a0', 'dist_C_1_a0', 'dist_C_2_a0', 'dist_C_3_a0', 'dist_C_4_a0', 'dist_F_0_a0', 'dist_F_1_a0',
        'dist_F_2_a0', 'dist_F_3_a0', 'dist_F_4_a0', 'dist_H_0_a0', 'dist_H_1_a0', 'dist_H_2_a0', 'dist_H_3_a0',
        'dist_H_4_a0', 'dist_N_0_a0', 'dist_N_1_a0', 'dist_N_2_a0', 'dist_N_3_a0', 'dist_N_4_a0', 'dist_O_0_a0',
        'dist_O_1_a0', 'dist_O_2_a0', 'dist_O_3_a0', 'dist_O_4_a0',
        'EN_a0', 'rad_a0', 'n_bonds_a0', 'bond_lengths_mean_a0', 'bond_lengths_std_a0', 'bond_lengths_median_a0',

        'dist_C_0_a1', 'dist_C_1_a1', 'dist_C_2_a1', 'dist_C_3_a1', 'dist_C_4_a1', 'dist_F_0_a1', 'dist_F_1_a1',
        'dist_F_2_a1', 'dist_F_3_a1', 'dist_F_4_a1', 'dist_H_0_a1', 'dist_H_1_a1', 'dist_H_2_a1', 'dist_H_3_a1',
        'dist_H_4_a1', 'dist_N_0_a1', 'dist_N_1_a1', 'dist_N_2_a1', 'dist_N_3_a1', 'dist_N_4_a1', 'dist_O_0_a1',
        'dist_O_1_a1', 'dist_O_2_a1', 'dist_O_3_a1', 'dist_O_4_a1',
        'EN_a1', 'rad_a1', 'n_bonds_a1', 'bond_lengths_mean_a1', 'bond_lengths_std_a1', 'bond_lengths_median_a1',

        # Criskiev
        'atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'd_2_0', 'd_2_1', # 'd_1_0'
        'd_3_0', 'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0', 'd_5_1', 'd_5_2', 'd_5_3', 'd_6_0',
        'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1', 'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0',
        'd_9_1', 'd_9_2', 'd_9_3',

        # Criskiev extra
        # 'd_1_0_log', 'd_2_0_log', 'd_2_1_log', 'd_3_0_log', 'd_3_1_log', 'd_3_2_log', 'd_4_0_log', 'd_4_1_log',
        # 'd_4_2_log', 'd_4_3_log', 'd_5_0_log', 'd_5_1_log', 'd_5_2_log', 'd_5_3_log', 'd_6_0_log', 'd_6_1_log',
        # 'd_6_2_log', 'd_6_3_log', 'd_7_0_log', 'd_7_1_log', 'd_7_2_log', 'd_7_3_log', 'd_8_0_log', 'd_8_1_log',
        # 'd_8_2_log', 'd_8_3_log', 'd_9_0_log', 'd_9_1_log', 'd_9_2_log', 'd_9_3',
        #
        # 'd_1_0_recp', 'd_2_0_recp', 'd_2_1_recp', 'd_3_0_recp', 'd_3_1_recp', 'd_3_2_recp', 'd_4_0_recp', 'd_4_1_recp',
        # 'd_4_2_recp', 'd_4_3_recp', 'd_5_0_recp', 'd_5_1_recp', 'd_5_2_recp', 'd_5_3_recp', 'd_6_0_recp', 'd_6_1_recp',
        # 'd_6_2_recp', 'd_6_3_recp', 'd_7_0_recp', 'd_7_1_recp', 'd_7_2_recp', 'd_7_3_recp', 'd_8_0_recp', 'd_8_1_recp',
        # 'd_8_2_recp', 'd_8_3_recp', 'd_9_0_recp', 'd_9_1_recp', 'd_9_2_recp', 'd_9_3'
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


# def t5_criskiev_features_extra(train, test):
#     def _helper(df):
#         criskiev_columns = ['d_1_0', 'd_2_0', 'd_2_1', 'd_3_0', 'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3',
#                             'd_5_0', 'd_5_1', 'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3', 'd_7_0', 'd_7_1',
#                             'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2', 'd_8_3', 'd_9_0', 'd_9_1', 'd_9_2', 'd_9_3']
#         for c in criskiev_columns:
#             # df[f'{c}_log'] = np.log(df[c])
#             # df[f'{c}_log'] = df[f'{c}_log'].astype('float32')
#             # df[f'{c}_recp'] = 1 / (df[c] ** 3)
#             # df[f'{c}_recp'] = df[f'{c}_recp'].astype('float32')
#             df[c] = 1 / (df[c] ** 3)
#             df[c] = df[c].astype('float32')
#
#     _helper(train)
#     _helper(test)


def t5_to_parquet(work_dir, train, test, structures, contributions):
    train.to_parquet(f'{work_dir}/t5_train.parquet')
    test.to_parquet(f'{work_dir}/t5_test.parquet')
    structures.to_parquet(f'{work_dir}/t5_structures.parquet')
    contributions.to_parquet(f'{work_dir}/t5_contributions.parquet')


def t5_read_parquet(work_dir):
    train = pd.read_parquet(f'{work_dir}/t5_train.parquet')
    test = pd.read_parquet(f'{work_dir}/t5_test.parquet')
    structures = pd.read_parquet(f'{work_dir}/t5_structures.parquet')
    contributions = pd.read_parquet(f'{work_dir}/t5_contributions.parquet')

    return train, test, structures, contributions


def t5_do_predict(train, test, TYPE_WL, TARGET_WL, PARAMS, N_FOLD, N_ESTIMATORS, SEED, X, X_test, labels, subtype_col = None):
    for type_name in TYPE_WL:
        _PARAMS = {**PARAMS['_'], **PARAMS[type_name]} if type_name in PARAMS.keys() else PARAMS['_']
        _N_FOLD = N_FOLD[type_name] if type_name in N_FOLD.keys() else N_FOLD['_']
        t = labels['type'].transform([type_name])[0]

        for target in TARGET_WL:
            _N_ESTIMATORS = N_ESTIMATORS[type_name] if type_name in N_ESTIMATORS.keys() else N_ESTIMATORS['_']
            _N_ESTIMATORS = _N_ESTIMATORS if target != 'fc' else _N_ESTIMATORS * 3
            score_accumulator = []

            subtype_col = subtype_col if subtype_col != None else 'qcut_subtype_0'

            subtypes = np.sort(train.loc[(train['type'] == t), subtype_col].unique())
            for st in subtypes:
                X_t = X.loc[(X['type'] == t) & (X[subtype_col] == st)]
                X_test_t = X_test.loc[(X_test['type'] == t) & (X_test[subtype_col] == st)]
                y_t = train.loc[(train['type'] == t) & (train[subtype_col] == st), target]

                folds = KFold(n_splits=_N_FOLD, shuffle=True, random_state=SEED)

                print("Training of type %s, component '%s', subtype: (%d/%d) train size: %d" % (type_name, target, st, len(subtypes), len(y_t)))

                result_dict_lgb_oof = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=_PARAMS, folds=folds,
                                                             model_type='lgb', eval_metric='group_mae',
                                                             plot_feature_importance=True,
                                                             verbose=100, early_stopping_rounds=200,
                                                             n_estimators=_N_ESTIMATORS)

                train.loc[(train['type'] == t) & (train[subtype_col] == st), f'oof_{target}'] = result_dict_lgb_oof['oof']
                test.loc[(test['type'] == t) & (test[subtype_col] == st), f'oof_{target}'] = result_dict_lgb_oof['prediction']
                score_accumulator = score_accumulator + list(result_dict_lgb_oof['scores'])

            print('CV mean score [%s, %s]: %.4f, std: %.4f' % (type_name, target, np.mean(score_accumulator), np.std(score_accumulator)))


def t5_learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


def t5_learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3


def t5_learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


def t5_learning_rate_005_decay_power_099931(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate * np.power(.99931, current_iter)
    return lr if lr > 1e-3 else 1e-3


def t5_learning_rate_020_decay_power_099965(current_iter):
    base_learning_rate = 0.2
    lr = base_learning_rate * np.power(.99965, current_iter)
    return max(lr, 3e-3)

##### COPY__PASTE__LIB__END #####

# Author: Sayantan Majumdar
# Email: smxnv@mst.edu


import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGCV
from dask_ml.model_selection import RandomizedSearchCV as DaskRCV
from dask_jobqueue import SLURMCluster
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, RandomizedSearchCV, cross_validate, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .sysops import makedirs, make_proper_dir_name


def get_model_param_dict(random_state=0):
    """
    Get model object dictionaries and parameter dictionary for different models
    :param random_state: PRNG seed
    :return: Dictionary of the model objects and Dictionary of models containing dictionary of the corresponding
    hyperparameters as tuples
    """

    model_dict = {
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True
        ),
        'BRF': LGBMRegressor(
            boosting_type='rf', tree_learner='feature',
            bagging_freq=1, random_state=random_state,
            deterministic=True, force_row_wise=True
        ),
        'RF': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        'DT': DecisionTreeRegressor(random_state=random_state),
        'BT': BaggingRegressor(random_state=random_state, n_jobs=-1),
        'ABR': AdaBoostRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(n_jobs=-1),
        'LR': LinearRegression(n_jobs=-1)
    }

    param_dict = {'LGBM': {
        'n_estimators': [500, 700, 900],
        'boosting_type': ['gbdt', 'goss'],
        'max_depth': [20, -1],
        'learning_rate': [0.01, 0.05],
        'subsample': [1, 0.9],
        'colsample_bytree': [1, 0.9],
        'colsample_bynode': [1, 0.9],
        'path_smooth': [0.1, 0.2, 0.3],
        'num_leaves': [31, 63],
        'min_data_in_leaf': [20, 30, 35]
    }, 'BRF': {
        'n_estimators': [500, 700, 900],
        'max_depth': [20, 25, 30, -1],
        'learning_rate': [0.006, 0.007, 0.008, 0.009],
        'bagging_fraction': [0.5, 0.4, 0.3],
        'path_smooth': [0.1, 0.2, 0.3],
        'num_leaves': [127, 255, 511],
        'min_data_in_leaf': [10, 20, 30, 50]
    }, 'RF': {
        'n_estimators': [300, 400, 500, 600, 700, 800],
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, 20, None],
        'max_samples': [None, 0.9, 0.8, 0.7],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'ETR': {
        'n_estimators': [300, 400, 500, 600, 700, 800],
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, 20, None],
        'max_samples': [None, 0.9, 0.8, 0.7],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'DT': {
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, 20, None],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'BT': {
        'n_estimators': [300, 400, 500, 600, 700],
        'max_features': [5, 6, 7],
        'max_samples': [None, 0.9, 0.8, 0.7]
    }, 'ABR': {
        'n_estimators': [300, 400, 500, 600, 700],
        'learning_rate': [0.005, 0.0098, 0.01, 0.05],
        'loss': ['linear', 'square']
    }, 'KNN': {
        'n_neighbors': [5, 8, 10],
        'weights': ['uniform', 'distance'],
        'leaf_size': [30, 50, 20],
        'p': [1, 2, 3, 5],
    }, 'LR': {
    }}
    return model_dict, param_dict


def build_crop_ml_models(x_train, x_test, y_train, y_test, year_train, year_test, year_col, crop_col, model_dir,
                         model_name='XGB', random_state=43, load_model=False, fold_count=5, repeats=3, x_scaler=None,
                         y_scaler=None, randomized_search=False, test_size=0.2, shuffle_split=False, use_dask=False):
    """
    Build individual ML models for each crop type
    :param x_train: X_train numpy array or pandas dataframe
    :param x_test: X_test numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param y_test: y_test numpy array
    :param year_train: Year train data frame to append to train data
    :param year_test: Year test data frame to append to test data
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column
    :param model_dir: Model directory to store/load model
    :param model_name: ML model name as per the model_dict keys
    :param random_state: PRNG seed
    :param load_model: Set model name to load existing model
    :param fold_count: Number of folds for KFold
    :param repeats: Number of repeats for KFold
    :param x_scaler: X scaler object
    :param y_scaler: y scaler object
    :param randomized_search: Set True to use the more computationally efficient RandomizedSearchCV
    :param test_size: Test data size for ShuffleSplit, works only when shuffle_split is True
    :param shuffle_split: Set True to use ShuffleSplit instead of KFold
    :param use_dask: Flag for using dask
    :return: List of trained models
    """

    crop_train_arr = x_train[crop_col].to_numpy().ravel()
    crop_test_arr = x_test[crop_col].to_numpy().ravel()
    models = []
    merged_pred_df = pd.DataFrame()
    for crop in x_train[crop_col].unique():
        crop_train_check = x_train[crop_col] == crop
        crop_test_check = x_test[crop_col] == crop
        x_train_data = x_train[crop_train_check]
        x_test_data = x_test[crop_test_check]
        x_train_data = x_train_data.drop(columns=[crop_col])
        x_test_data = x_test_data.drop(columns=[crop_col])
        y_train_data = y_train[crop_train_arr == crop]
        y_test_data = y_test[crop_test_arr == crop]
        year_train_df = year_train[crop_train_check]
        year_test_df = year_test[crop_test_check]
        crop_model_name = model_name + '_' + str(crop)
        model = build_ml_model(x_train_data, y_train_data, model_dir, crop_model_name, random_state, load_model,
                               fold_count, repeats, y_scaler, randomized_search, test_size, shuffle_split, use_dask)
        models.append(model)
        pred_df = get_prediction_results(model, x_train_data, x_test_data, y_train_data, y_test_data, x_scaler,
                                         y_scaler, year_train_df, year_test_df, model_dir, crop_model_name, year_col,
                                         crop_col)
        merged_pred_df = merged_pred_df.append(pred_df)
    calc_train_test_metrics(merged_pred_df, crop_col, year_col)
    merged_pred_df.to_csv(model_dir + 'Merged_Crop_Predictions_{}.csv'.format(model_name), index=False)
    return models


def build_ml_model(x_train, y_train, model_dir, model_name='XGB', random_state=43, load_model=False,
                   fold_count=5, repeats=3, y_scaler=None, randomized_search=False, test_size=0.2, shuffle_split=False,
                   use_dask=False):
    """
    Build an ML model
    :param x_train: X_train numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param model_dir: Model directory to store/load model
    :param model_name: ML model name as per the model_dict keys
    :param random_state: PRNG seed
    :param load_model: Set model name to load existing model
    :param fold_count: Number of folds for KFold
    :param repeats: Number of repeats for KFold
    :param y_scaler: y scaler object
    :param randomized_search: Set True to use the more computationally efficient RandomizedSearchCV
    :param test_size: Test data size for ShuffleSplit, works only when shuffle_split is True
    :param shuffle_split: Set True to use ShuffleSplit instead of KFold
    :param use_dask: Flag for using dask
    :return: Trained model object
    """

    model_file = model_dir + model_name
    if not load_model:
        dask_client = None
        cv_lib = 'sklearn'
        if use_dask:
            cluster = SLURMCluster(
                cores=64,
                processes=1,
                memory="10G",
                walltime="00:30:00",
                interface='ib0',
                env_extra=['#SBATCH --out=Foundry-Dask-%j.out']
            )
            cluster.adapt(
                minimum=10, maximum=50,
                minimum_jobs=10, maximum_jobs=50,
                minimum_memory='8G', maximum_memory='10G'
            )
            dask_client = Client(cluster)
            print('Waiting for dask workers...')
            dask_client.wait_for_workers(10)
            cv_lib = 'dask_ml'
        model_dict, param_dict = get_model_param_dict(random_state)
        if not shuffle_split:
            cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        else:
            cv = ShuffleSplit(n_splits=fold_count, test_size=test_size, random_state=random_state)
        makedirs([make_proper_dir_name(model_dir)])
        x = x_train.copy()
        y = y_train.copy()
        print('\nSearching best params for {}...'.format(model_name))
        if '_' in model_name:
            model_name = model_name[: model_name.find('_')]
        model = model_dict[model_name]
        scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        cv_func_dict = {
            'dask_ml': {1: DaskRCV, 0: DaskGCV},
            'sklearn': {1: RandomizedSearchCV, 0: GridSearchCV}
        }
        cv_func = cv_func_dict[cv_lib][int(randomized_search)]
        if randomized_search:
            model_grid = cv_func(
                estimator=model, param_distributions=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True, random_state=random_state
            )
        else:
            model_grid = cv_func(
                estimator=model, param_grid=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True
            )
        model_grid.fit(x, y)
        get_grid_search_stats(model_grid, y_scaler)
        model = model_grid.best_estimator_
        print('Best params: ', model_grid.best_params_)
        pickle.dump(model, open(model_file, mode='wb+'))
        if dask_client:
            dask_client.close()
    else:
        model = pickle.load(open(model_file, mode='rb'))
    if model_name in ['RF', 'ETR', 'LGBM', 'BRF']:
        imp_dict = {'Features': list(x_train.columns)}
        f_imp = np.array(model.feature_importances_).astype(np.float)
        if model_name in ['LGBM', 'BRF']:
            f_imp /= np.sum(f_imp)
        imp_dict['F_IMP'] = np.round(f_imp, 5)
        imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
        print(imp_df)
        imp_df.to_csv(model_dir + 'F_IMP.csv', index=False)
    return model


def calc_train_test_metrics(pred_df, crop_col, year_col):
    """
    Calculate train and test metrics from the prediction data frames
    :param pred_df: Prediction data frame
    :param crop_col: Name of the crop column
    :param year_col: Name of the year column
    :return: None
    """

    train_data = pred_df[pred_df.DATA == 'TRAIN']
    test_data = pred_df[pred_df.DATA == 'TEST']
    train_actual = train_data.Actual_AF.to_numpy().ravel()
    train_pred = train_data.Pred_AF.to_numpy().ravel()
    test_actual = test_data.Actual_AF.to_numpy().ravel()
    test_pred = test_data.Pred_AF.to_numpy().ravel()
    print('***Overall stats***\n')
    print('Train + Validation results...')
    r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('\nTest results...')
    r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    cols = [col for col in pred_df.columns if col.startswith(crop_col)] + [year_col]
    for col in cols:
        print('\n***{} specific stats***\n'.format(col))
        if col != year_col:
            col_val_list = [1]
        else:
            col_val_list = np.unique(np.append(train_data[col].unique(), test_data[col].unique()))
        for val in sorted(col_val_list):
            print('\n{} type: {}'.format(col, val))
            train_actual = train_data[train_data[col] == val]['Actual_AF'].to_numpy().ravel()
            train_pred = train_data[train_data[col] == val]['Pred_AF'].to_numpy().ravel()
            test_actual = test_data[test_data[col] == val]['Actual_AF'].to_numpy().ravel()
            test_pred = test_data[test_data[col] == val]['Pred_AF'].to_numpy().ravel()
            print('Train + Validation results...')
            r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
            print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
            print('\nTest results...')
            r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
            print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_grid_search_stats(gs_model, y_scaler=None):
    """
    Get GridSearchCV stats
    :param gs_model: Fitted GridSearchCV/RandomizedSearchCV object
    :param y_scaler:y scaler object
    :return:
    """

    scores = gs_model.cv_results_
    print('Train Results...')
    r2 = scores['mean_train_r2'].mean()
    rmse = np.sqrt(-scores['mean_train_neg_mean_squared_error'].mean())
    mae = -scores['mean_train_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('Validation Results...')
    r2 = scores['mean_test_r2'].mean()
    rmse = np.sqrt(-scores['mean_test_neg_mean_squared_error'].mean())
    mae = -scores['mean_test_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_test_cv_stats(model, x_test, y_test, y_scaler, fold_count=5, repeats=3, random_state=0):
    """
    Get cross validation statistics for the test data
    :param model: Model object
    :param x_test: X_test numpy array or pandas dataframe
    :param y_test: y_test numpy array
    :param fold_count: Number of folds for KFold
    :param repeats: Number of repeats for KFold
    :param y_scaler: y scaler object
    :param random_state: PRNG seed
    :return: None
    """

    cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
    x_test = np.ascontiguousarray(x_test)
    print('Test Results...')
    scores = cross_validate(model, x_test, y_test, cv=cv,
                            scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'),
                            return_train_score=True, n_jobs=-1)
    r2 = scores['test_r2'].mean()
    rmse = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
    mae = -scores['test_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_prediction_stats(actual_values, pred_values, precision=3):
    """
    Get prediction statistics R^2, MAE, RMSE
    :param actual_values: Numpy array of actual values
    :param pred_values: Numpy array of predicted values
    :param precision: Floating point precision to use
    :return: R^2, MAE, RMSE as tuples
    """

    r2, mae, rmse = (np.nan,) * 3
    if actual_values.size and pred_values.size:
        r2 = np.round(r2_score(actual_values, pred_values), precision)
        mae = np.round(mean_absolute_error(actual_values, pred_values), precision)
        rmse = np.round(mean_squared_error(actual_values, pred_values, squared=False), precision)
    return r2, mae, rmse


def get_prediction_results(model, x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test,
                           model_dir, model_name='BRF', year_col='ReportYear', crop_col='Crop_CDL'):
    """
    Get model prediction results
    :param model: Trained model object
    :param x_train: X_train numpy array or pandas dataframe
    :param x_test: X_test numpy array or pandas dataframe
    :param y_train: y_train numpy array
    :param y_test: y_test numpy array
    :param x_scaler: X scaler object
    :param y_scaler: y scaler object
    :param year_train: Year train data frame to append to train data
    :param year_test: Year test data frame to append to test data
    :param model_dir: Model directory to store/load results
    :param model_name: Model name
    :param year_col: Name of the year column
    :param crop_col: Name of the crop column
    :return: Prediction data frame
    """

    y_pred_train = np.abs(model.predict(x_train))
    y_pred_test = np.abs(model.predict(x_test))
    if x_scaler and y_scaler:
        x_train = pd.DataFrame(x_scaler.inverse_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(x_scaler.inverse_transform(x_test), columns=x_test.columns)
        y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    train_df = x_train.copy()
    train_df[year_col] = year_train[year_col].to_numpy().ravel()
    test_df = x_test.copy()
    test_df[year_col] = year_test[year_col].to_numpy().ravel()
    train_df['DATA'] = ['TRAIN'] * train_df.shape[0]
    train_df['Pred_AF'] = y_pred_train
    train_df['Actual_AF'] = y_train
    test_df['DATA'] = ['TEST'] * test_df.shape[0]
    test_df['Pred_AF'] = y_pred_test
    test_df['Actual_AF'] = y_test
    pred_df = train_df.append(test_df)
    pred_df['Error_AF'] = pred_df['Actual_AF'] - pred_df['Pred_AF']
    if '_' in model_name:
        crop = model_name[model_name.find('_') + 1:]
        pred_df[crop_col] = [crop] * pred_df['Error_AF'].size
    pred_df.to_csv(model_dir + 'Predictions_{}.csv'.format(model_name), index=False)
    return pred_df

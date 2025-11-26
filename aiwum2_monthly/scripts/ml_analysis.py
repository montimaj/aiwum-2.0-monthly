# Perform feature importance and Permutation Importance analysis on the AIWUM2 dataset

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import calendar
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error


def normalized_rmse(
        y: np.array,
        y_pred: np.array
) -> float:
    """
    Normalized RMSE using mean.

    Args:
        y (np.array): Actual values.
        y_pred (np.array): Predicted values.

    Returns:
        float: Normalized RMSE using mean.
    """

    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    nrmse = root_mean_squared_error(
        y, y_pred
    ) * 100 / mean_y
    return nrmse


def generate_imp_plots(
    model_dir: str,
    model_name: str,
    x_train_csv: str,
    y_train_csv: str,
    x_test_csv: str,
    y_test_csv: str,
    output_dir: str
) -> None:
    """
    Generate feature importance and permutation importance plots.
    Args:
        model_dir (str): Model directory.
        model_name (str): Name of the model.
        x_train_csv (str): Training data predictor CSV.
        y_train_csv (str): Training data label CSV.
        x_test_csv (str): Test data predictor CSV.
        y_test_csv (str): Test data label CSV.
        output_dir (str): Output directory.

    Returns:
        None.
    """

    ml_model = pickle.load(open(f'{model_dir}{model_name}', 'rb'))
    fimp_df = pd.read_csv(f'{model_dir}F_IMP.csv').head(5)
    x_train_df = pd.read_csv(x_train_csv)
    y_train_df = pd.read_csv(y_train_csv)
    x_test_df = pd.read_csv(x_test_csv)
    y_test_df = pd.read_csv(y_test_csv)

    # Feature importance plot
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 16})
    sns.barplot(x='F_IMP', y='Features', data=fimp_df)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.savefig(f'{output_dir}Feature_Importance.svg', dpi=600, bbox_inches='tight')

    # calculate permutation importance
    perm_scorer = make_scorer(normalized_rmse, greater_is_better=False)
    train_result = permutation_importance(
        ml_model, x_train_df.to_numpy(), y_train_df,
        n_repeats=10, random_state=1,
        n_jobs=-1, scoring=perm_scorer
    )
    test_result = permutation_importance(
        ml_model, x_test_df.to_numpy(), y_test_df,
        n_repeats=10, random_state=1,
        n_jobs=-1, scoring=perm_scorer
    )
    sorted_importances_idx = train_result.importances_mean.argsort()
    train_importances = pd.DataFrame(
        train_result.importances[sorted_importances_idx].T,
        columns=x_train_df.columns[sorted_importances_idx],
    )
    test_importances = pd.DataFrame(
        test_result.importances[sorted_importances_idx].T,
        columns=x_train_df.columns[sorted_importances_idx],
    )
    new_colnames = {
        'SM_IDAHO': 'Soil Moisture',
        'SSEBop': 'ET',
        'tmax': 'TMAX',
        'crop_Rice': 'Rice',
        'crop_Fish Culture': 'Aquaculture',
        'lat_dd': 'Latitude',
        'long_dd': 'Longitude',
        'Month_8': 'Aug',
        'Month_9': 'Sep',
        'RO': 'Runoff'
    }
    train_importances = train_importances.rename(columns=new_colnames)
    test_importances = test_importances.rename(columns=new_colnames)
    train_importances = train_importances.iloc[:, -10:]
    test_importances = test_importances.iloc[:, -10:]

    for name, importances in zip(["train", "test"], [train_importances, test_importances]):
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 16})
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_xlabel("Increase in RMSE (%)")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.figure.tight_layout()
        plt.savefig(f'{output_dir}{model_name}_{name}_PI.svg', dpi=600)
        plt.clf()


def plot_train_test_scatter(
    predictions_csv: str,
    output_dir: str
) -> None:
    """
    Plot train and test scatter plots.
    Args:
        predictions_csv (str): Predictions CSV file path.
        output_dir (str): Output directory.

    Returns:
        None.
    """

    predictions_df = pd.read_csv(predictions_csv)
    predictions_df['Month'] = predictions_df['Month'].apply(lambda x: calendar.month_abbr[x])
    predictions_df_copy = predictions_df.copy(deep=True)
    predictions_df['Month'] = pd.Categorical(
        predictions_df['Month'],
        categories=calendar.month_abbr[1:],
        ordered=True
    )
    train_data = predictions_df_copy[predictions_df.DATA == 'TRAIN']
    test_data = predictions_df[predictions_df.DATA == 'TEST']
    min_train = train_data['Actual_GW'].min()
    max_train = train_data['Actual_GW'].max()
    min_test = test_data['Actual_GW'].min()
    max_test = test_data['Actual_GW'].max()

    # These are printed on the console. See: ../../AIWUM2_Data/Outputs/Models/Params_LGBM
    train_score_dict = {
        'R2': 0.85,
        'RMSE (mm/month)': 14.79,
        'MAE (mm/month)': 6.04
    }
    val_score_dict = {
        'R2': 0.72,
        'RMSE (mm/month)': 20.57,
        'MAE (mm/month)': 7.99
    }
    test_score_dict = {
        'R2': 0.73,
        'RMSE (mm/month)': 20.21,
        'MAE (mm/month)': 7.6
    }

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 16})
    s = sns.scatterplot(
        data=train_data,
        x='Pred_GW',
        y='Actual_GW',
        color='black',
        alpha=0.5,
        legend="full"
    )
    # plot 1:1 line for train and test data
    sns.lineplot(
        x=[min_train, max_train],
        y=[min_train, max_train],
        color='red',
        linestyle='--',
        ax=s,
        legend="full"
    )
    plt.xlim(0, 850)
    plt.ylim(0, 850)
    plt.ylabel('Actual Water Use (mm)')
    plt.xlabel('Predicted Water Use (mm)')

    plt.savefig(f'{output_dir}{model_name}_Train_Scatter.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 16})
    s = sns.scatterplot(
        data=test_data,
        x='Pred_GW',
        y='Actual_GW',
        color='black',
        alpha=0.5,
        legend="full"
    )
    sns.lineplot(
        x=[min_test, max_test],
        y=[min_test, max_test],
        color='red',
        linestyle='--',
        ax=s,
        legend="full"
    )
    plt.xlim(0, 850)
    plt.ylim(0, 850)
    plt.ylabel('Actual Water Use (mm)')
    plt.xlabel('Predicted Water Use (mm)')
    plt.savefig(f'{output_dir}{model_name}_Test_Scatter.png', dpi=300, bbox_inches='tight')
    plt.clf()

    gs = calendar.month_abbr[4: 10]
    predictions_df_gs = predictions_df_copy[predictions_df_copy.Month.isin(gs)].copy()
    predictions_df_gs['Month'] = pd.Categorical(
        predictions_df_gs['Month'],
        categories=calendar.month_abbr[4: 10],
        ordered=True
    )
    month_colors = sns.color_palette('tab10', n_colors=6)
    train_data = predictions_df_gs[predictions_df_gs.DATA == 'TRAIN'].copy()
    test_data = predictions_df_gs[predictions_df_gs.DATA == 'TEST'].copy()
    train_data['Std_Residuals'] = train_data['Error_GW'] / train_data['Error_GW'].std()
    test_data['Std_Residuals'] = test_data['Error_GW'] / test_data['Error_GW'].std()
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 16})
    sns.kdeplot(
        data=train_data,
        x='Std_Residuals',
        fill=True,
        hue='Month',
        palette=month_colors,
        hue_order=calendar.month_abbr[4: 10],
        common_norm=False
    )
    plt.xlim(-3, 3)
    plt.xlabel('Standardized Residuals')
    plt.savefig(f'{output_dir}{model_name}_Train_Residuals.svg', dpi=600, bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 16})
    sns.kdeplot(
        data=test_data,
        x='Std_Residuals',
        fill=True,
        hue='Month',
        palette=month_colors,
        hue_order=calendar.month_abbr[4: 10],
        common_norm=False
    )
    plt.xlim(-3, 3)
    plt.xlabel('Standardized Residuals')
    plt.savefig(f'{output_dir}{model_name}_Test_Residuals.svg', dpi=600, bbox_inches='tight')


def aiwum_comparison(aiwum_comp_csv: str) -> None:
    """
    Compare AIWUM2.0 and AIWUM 1.1.
    Args:
        aiwum_comp_csv (str): AIWUM comparison CSV file.

    Returns:
        None.
    """
    aiwum_comp_df = pd.read_csv(aiwum_comp_csv)
    aiwum_comp_df = aiwum_comp_df[aiwum_comp_df.Month != 10].drop(columns=['Year'])
    aiwum_comp_df = aiwum_comp_df.groupby('Month').mean().reset_index()
    aiwum_comp_df['Diff'] = aiwum_comp_df['AIWUM1.1'] - aiwum_comp_df['AIWUM2.1']
    aiwum_comp_df['Diff_Percent'] = aiwum_comp_df['Diff'] * 100 / aiwum_comp_df['AIWUM1.1']
    aiwum_comp_df['Diff_Percent'] = aiwum_comp_df['Diff_Percent'].abs()
    print(aiwum_comp_df)
    
    
def plot_training_data(map_csv: str, output_dir: str) -> None:
    """
    Plot water use distributions.
    Args:
        map_csv: Cleaned CSV file.
        output_dir: Output directory.

    Returns:
        None.
    """
    
    map_df = pd.read_csv(map_csv)
    map_df = map_df[map_df.Month.isin(range(4, 10))]
    map_df['Month'] = pd.Categorical(
        map_df['Month'],
        categories=range(4, 10),
        ordered=True
    )
    for crop in ['Corn', 'Cotton', 'Rice', 'Soybeans', 'Fish Culture']:
        # show 6 months as subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        for i, month in enumerate(range(4, 10)):
            ax = axs[i]
            month_data = map_df[(map_df.Month == month) & (map_df.crop == crop)]
            sns.boxplot(
                data=month_data,
                x='AF_Acre',
                ax=ax
            )
            ax.set_title(f'{calendar.month_abbr[month]}')
            ax.set_xlabel('Groundwater Use (mm)')
            ax.set_ylabel('')
        crop_name = 'Aquaculture' if crop == 'Fish Culture' else crop
        plt.savefig(f'{output_dir}{crop_name}_GW_Distribution.png', dpi=300, bbox_inches='tight')

    # show the combined distribution for corn, cotton, rice, soybeans for each month
    for month in range(4, 10):
        month_data = map_df[
            (map_df.Month == month) & (map_df.crop.isin(['Corn', 'Cotton', 'Rice', 'Soybeans']))
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=month_data,
            x='AF_Acre',
            hue='crop',
            ax=ax
        )
        if month == 4:
            ax.set_xlim(-1e-5, month_data['AF_Acre'].mean() + 0.01 * month_data['AF_Acre'].std())
            # print mean crop wu
            print(month_data.groupby('crop')['AF_Acre'].mean())
        ax.set_title(f'{calendar.month_abbr[month]}')
        ax.set_xlabel('Groundwater Use (mm)')
        ax.set_ylabel('')
        plt.savefig(f'{output_dir}{calendar.month_abbr[month]}_GW_Distribution.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    model_dir = '../../AIWUM2_Data/Outputs/Models/'
    model_name = 'LGBM'
    x_train_csv = '../../AIWUM2_Data/Outputs/X_train.csv'
    y_train_csv = '../../AIWUM2_Data/Outputs/y_train.csv'
    x_test_csv = '../../AIWUM2_Data/Outputs/X_test.csv'
    y_test_csv = '../../AIWUM2_Data/Outputs/y_test.csv'
    predictions_csv = '../../AIWUM2_Data/Outputs/Models/Predictions_LGBM.csv'
    output_dir = 'Script_Outputs/ML_Analysis/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # generate_imp_plots(
    #     model_dir,
    #     model_name,
    #     x_train_csv,
    #     y_train_csv,
    #     x_test_csv,
    #     y_test_csv,
    #     output_dir
    # )
    # plot_train_test_scatter(
    #     predictions_csv,
    #     output_dir
    # )
    # aiwum_comp_csv = '../../AIWUM2_Data/Outputs/AIWUM_Comparison/Annual_Tot_AIWUM.csv'
    # aiwum_comparison(aiwum_comp_csv)
    map_csv = '../../AIWUM2_Data/Outputs/Cleaned_MAP_GW_Data.csv'
    plot_training_data(map_csv, output_dir)

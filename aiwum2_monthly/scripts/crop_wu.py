"""
Independent script for generating annual crop-specific water use rasters at 100 m and 1 km spatial resolutions (2014-2021).
"""

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu


import sys
import pandas as pd
import seaborn as sns
import calendar
import os
import rasterio as rio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import swifter
import dask.dataframe as dd
from rasterio.mask import mask
from joblib import Parallel, delayed
from os.path import dirname, abspath
from glob import glob
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from aiwum2_monthly.maplibs.sysops import makedirs
from aiwum2_monthly.maplibs.rasterops import read_raster_as_arr, reproject_raster_gdal


def get_crop_wu_df(
    year_month: tuple[int, str],
    aiwum2_dir: str,
    cdl_dir: str
) -> pd.DataFrame:
    """
    Use this function to get monthly dataframes.

    Args:
        year_month (tuple[int, str]): Year and month.
        aiwum2_dir (str): AIWUM 2.0 directory.
        cdl_dir (str): CDL directory.
    Returns:
        A pandas dataframe.
    """
    year, month = year_month
    aiwum2_file = glob(f'{aiwum2_dir}*{year}_{month}.tif')[0]
    cdl_file = glob(f'{cdl_dir}*{year}.tif')[0]
    aiwum2_arr = read_raster_as_arr(aiwum2_file, get_file=False).ravel()
    cdl_arr = read_raster_as_arr(cdl_file, get_file=False).ravel()
    arr_size = aiwum2_arr.size
    crop_wu_dict = {
        'Year': [year] * arr_size,
        'Month': [month] * arr_size,
        'Crop': cdl_arr.tolist(),
        'WU (m3)': aiwum2_arr.tolist()
    }
    crop_df = pd.DataFrame(crop_wu_dict).dropna()
    return crop_df


def get_crop_wu_mean_std(crop_wu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the mean and standard deviation of crop water use. 'Crop', 'Month', 'WU (m3)' columns are required.

    Args:
        crop_wu_df (pd.DataFrame): Crop water use dataframe.

    Returns:
        A pandas dataframe.
    """
    wu_col = 'WU (m3)'
    sd_col = 'SD_WU (m3)'
    crop_wu_df[sd_col] = crop_wu_df[wu_col].copy()
    crop_wu_df_monthly = crop_wu_df[['Crop', 'Month', wu_col, sd_col]].groupby([
        'Crop', 'Month',
    ]).agg({
        wu_col: 'mean',
        sd_col: 'std'
    }).reset_index()
    gs = calendar.month_abbr[4: 10]
    for col in [wu_col, sd_col]:
        crop_wu_df_monthly[col] = crop_wu_df_monthly.apply(
            lambda x: np.nan if (x.Crop != 'Aquaculture') and (x.Month not in gs) else x[col],
            axis=1
        )
    crop_wu_df_monthly = crop_wu_df_monthly.dropna()
    crop_wu_df_monthly['Month'] = pd.Categorical(
        crop_wu_df_monthly['Month'],
        categories=calendar.month_abbr[1:],
        ordered=True
    )
    crop_wu_df_monthly = crop_wu_df_monthly.sort_values(['Crop', 'Month']).reset_index(drop=True)
    return crop_wu_df_monthly


def create_crop_wu_csv(
    aiwum2_100m_dir: str,
    cdl_100m_dir: str,
    output_dir: str,
    start_month: int = 4,
    end_month: int = 9
) -> pd.DataFrame:
    """
    Create a CSV containing the monthly water use values and associated crop type
    Args:
        aiwum2_100m_dir (str): AIWUM 2.0 100 m predicted water use rasters.
        cdl_100m_dir (str): Corrected CDL 100 m rasters.
        output_dir (str): Output directory.
        start_month (int): Start month. Default is 4 (April).
        end_month (int): End month. Default is 9 (September).

    Returns:
        Monthly crop water use dataframe.
    """
    year_list = range(2014, 2022)
    output_file = f'{output_dir}AIWUM2_Crop_WU_Monthly_{year_list[0]}_{year_list[-1]}.parquet'
    if not os.path.exists(output_file):
        makedirs(output_dir)
        crop_type_dict = {
            1: 'Corn',
            2: 'Cotton',
            3: 'Rice',
            5: 'Soybeans',
            92: 'Aquaculture',
        }
        months = [calendar.month_abbr[m] for m in range(start_month, end_month + 1)]
        year_month = []
        cdl_dir = f'{output_dir}CDL/'
        makedirs(cdl_dir)
        for year in year_list:
            for idx, month in enumerate(months):
                year_month.append((year, month))
                if idx == 0:
                    aiwum2_file = glob(f'{aiwum2_100m_dir}*{year}_{month}.tif')[0]
                    cdl_file = glob(f'{cdl_100m_dir}*{year}.tif')[0]
                    cdl_cropped = f'{cdl_dir}{cdl_file.split(os.sep)[-1]}'
                    if not os.path.exists(cdl_cropped):
                        print('Cropping CDL', year, '...')
                        reproject_raster_gdal(
                            cdl_file,
                            cdl_cropped,
                            from_raster=aiwum2_file
                        )
        print('Building monthly dataframes...')
        crop_df_list = Parallel(n_jobs=-1)(
            delayed(get_crop_wu_df)(ym, aiwum2_100m_dir, cdl_dir) for ym in year_month
        )
        crop_wu_df = pd.concat(crop_df_list).reset_index(drop=True)
        crop_wu_df['Crop'] = crop_wu_df['Crop'].swifter.apply(
            lambda x: crop_type_dict[int(x)] if x in crop_type_dict.keys() else 'Other'
        )
        makedirs(output_dir)
        crop_wu_df.to_parquet(output_file, index=False)
    else:
        crop_wu_df = dd.read_parquet(output_file).compute()
    return crop_wu_df


def make_crop_wu_ts_plots(crop_wu_df_monthly: pd.DataFrame, output_dir: str) -> None:
    """
    Make crop water use time series plots.

    Args:
        crop_wu_df_monthly: Crop water use dataframe.
        output_dir: Output directory.

    Returns:
        None
    """
    print('Making crop water use time series plots...')
    plt.figure(figsize=(10, 5))
    unique_crops = sorted(crop_wu_df_monthly['Crop'].unique())
    crop_colors = sns.color_palette('tab10', n_colors=len(unique_crops))
    crop_wu_df_monthly['Month'] = pd.Categorical(
        crop_wu_df_monthly['Month'],
        categories=calendar.month_abbr[1:],
        ordered=True
    )
    avg_wu = crop_wu_df_monthly.groupby([
        'Crop', 'Month'
    ])['WU (m3)'].mean().reset_index().sort_values(by='Month')
    avg_wu.to_csv(f'{output_dir}AIWUM2_Avg_WU.csv', index=False)
    plt.rcParams.update({'font.size': 20})
    sns.lineplot(
        data=crop_wu_df_monthly,
        x='Month',
        y='WU (m3)',
        hue='Crop',
        marker='o',
        markersize=10,
        linewidth=2,
        palette=crop_colors,
        hue_order=unique_crops
    )

    # show standard deviation as error bars
    for i, crop in enumerate(crop_wu_df_monthly['Crop'].unique()):
        crop_data = crop_wu_df_monthly[crop_wu_df_monthly['Crop'] == crop]
        plt.rcParams.update({'font.size': 16})
        plt.errorbar(
            crop_data['Month'],
            crop_data['WU (m3)'],
            yerr=crop_data['SD_WU (m3)'],
            fmt='o',
            markersize=10,
            label=crop,
            capsize=5,
            capthick=2,
            color=crop_colors[i]
        )

    plt.ylabel('Mean Water Use ($m^3$)', fontsize=20)
    plt.xlabel('')

    # Save the plot
    plt.savefig(f'{output_dir}AIWUM2_Crop_WU_Monthly.svg', dpi=600, bbox_inches='tight')


def calc_crop_stats(
    cdl_dir: str,
    output_dir: str,
    region_file: str | None = None
) -> None:
    """
    Calculate crop statistics.
    Args:
        cdl_file: Corrected 100 m CDL file.
        region_file: MAP generalized regions file.
        output_dir: Output directory.

    Returns:
        None.
    """
    years = range(2014, 2022)
    crop_dict = {
        1: 'Corn',
        2: 'Cotton',
        3: 'Rice',
        5: 'Soybeans',
        92: 'Aquaculture',
        -1: 'Other'
    }
    crop_df = pd.DataFrame()
    for year in years:
        cdl_file = f'{cdl_dir}CDL_100m_Corrected_{year}.tif'
        cdl_arr = read_raster_as_arr(cdl_file, get_file=False).ravel()
        for crop in crop_dict.keys():
            if crop in cdl_arr:
                crop_count = (cdl_arr == crop).sum()
                df = pd.DataFrame({
                    'Year': [year],
                    'Crop': [crop_dict[crop]],
                    'Num_fields': [crop_count],
                })
                crop_df = pd.concat([crop_df, df])
    crop_df.to_csv(f'{output_dir}AIWUM2_Crop_Counts.csv', index=False)
    # print in scientific notation the average number of fields for each crop
    print(crop_df.groupby('Crop')['Num_fields'].mean().apply(lambda x: f'{x:.2e}'))

    # show the number of fields for each crop in each year as a bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=crop_df,
        x='Year',
        y='Num_fields',
        hue='Crop'
    )
    plt.ylabel('Number of Fields')
    plt.tight_layout()
    plt.savefig(f'{output_dir}AIWUM2_Crop_Counts.png', dpi=300)

    if region_file:
        regions_gdf = gpd.read_file(region_file)
        for year in years:
            cdl_df = pd.DataFrame()
            cdl_file = f'{cdl_dir}CDL_100_Corrected_{year}.tif'
            cdl_rio = rio.open(cdl_file)
            if year == 2014:
                regions_gdf = regions_gdf.to_crs(cdl_rio.crs)
            for idx, row in regions_gdf.iterrows():
                try:
                    cdl_mask, _ = mask(cdl_rio, [row.geometry], crop=True)
                    cdl_mask = cdl_mask[0].ravel()
                    cdl_mask = cdl_mask[cdl_mask != 0].astype(int)
                    df = pd.DataFrame({
                        'Region': [row.region] * cdl_mask.size,
                        'Year': [year] * cdl_mask.size,
                        'Crop': cdl_mask
                    })
                    cdl_df = pd.concat([cdl_df, df])
                except ValueError:
                    continue
            cdl_df = cdl_df.reset_index(drop=True)
            cdl_df['Crop'] = cdl_df['Crop'].swifter.apply(
                lambda x: crop_dict[int(x)] if x in crop_dict.keys() else 'Other'
            )

            # get percentage of each crop in each region
            crop_stats = cdl_df.groupby(['Region', 'Year', 'Crop']).size().unstack().reset_index()
            crops = crop_dict.values()
            crop_stats['Total'] = crop_stats[crops].sum(axis=1)
            for crop in crops:
                crop_stats[crop] *= 100 / crop_stats['Total']
            print(crop_stats)
            crop_stats.to_csv(f'{output_dir}AIWUM2_Crop_Stats_{year}.csv', index=False)


def make_distribution_plots(
    crop_wu_df: pd.DataFrame,
    output_dir: str,
    start_month: int = 4,
    end_month: int = 9
) -> None:
    """
    Make different distribution plots for crop water use, like KDE and boxplot.

    Args:
        crop_wu_df (pd.DataFrame): Crop water use dataframe.
        output_dir (str): Output directory.
        start_month (int): Start month. Default is 4 (April).
        end_month (int): End month. Default is 9 (September).

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    unique_crops = sorted(crop_wu_df['Crop'].unique())
    crop_colors = sns.color_palette('tab10', n_colors=len(unique_crops))
    gs = calendar.month_abbr[start_month: end_month + 1]
    crop_wu_df = crop_wu_df[crop_wu_df.Month.isin(gs)]
    print(crop_wu_df.Crop.value_counts())
    # sample 500 points from each crop for each month for each year
    crop_wu_sampled = crop_wu_df.groupby([
        'Crop', 'Month', 'Year'
    ]).sample(n=500, random_state=1).reset_index(drop=True)
    wu_col = 'WU (m3)'
    # show KDE plots for 12 months where each line represents a crop
    # Make 3X4 subplots
    print('Making KDE plots...')
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.ravel()
    for i, month in enumerate(calendar.month_abbr[start_month: end_month + 1]):
        ax = axs[i]
        plt.rcParams.update({'font.size': 16})
        sns.kdeplot(
            data=crop_wu_df[crop_wu_df['Month'] == month],
            x=wu_col,
            hue='Crop',
            fill=True,
            common_norm=False,
            palette=crop_colors,
            hue_order=unique_crops,
            ax=ax,
            legend=True if i == 0 else False
        )
        ax.set_xlim(-50, 2500)
        ax.set_title(month)
        ax.set_xlabel('Water Use ($m^3$)')
        ax.set_ylabel('Density')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'{output_dir}AIWUM2_Crop_WU_KDE.svg', dpi=600, bbox_inches='tight')

    # show the boxplot for each crop for each month
    print('Making boxplot...')
    crop_wu_sampled['Month'] = pd.Categorical(
        crop_wu_sampled['Month'],
        categories=calendar.month_abbr[start_month: end_month + 1],
        ordered=True
    )
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 20})
    sns.boxplot(
        data=crop_wu_sampled,
        x='Month',
        y=wu_col,
        hue='Crop',
        palette=crop_colors,
        hue_order=unique_crops
    )
    plt.ylabel('Water Use ($m^3$)')
    plt.xlabel('')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'{output_dir}AIWUM2_Crop_WU_Boxplot.svg', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    aiwum2_pred_dir = '../../AIWUM2_Data/Outputs/AIWUM2-1_Monthly_100m_m3_2014_2021/'
    cdl_dir = '../../AIWUM2_Data/Outputs/CDL_100m/'
    output_dir = 'Script_Outputs/Crop_WU/'
    start_month = 1
    end_month = 12
    # crop_wu_df = create_crop_wu_csv(
    #     aiwum2_pred_dir,
    #     cdl_dir,
    #     output_dir,
    #     start_month,
    #     end_month
    # )
    # crop_wu_monthly_df = get_crop_wu_mean_std(crop_wu_df)
    # make_crop_wu_ts_plots(crop_wu_monthly_df, output_dir)
    # make_distribution_plots(crop_wu_df, output_dir, start_month=4, end_month=9)
    map_region_file = '../../AIWUM2_Data/Inputs/MAP_generalized_regions/MAP_generalized_regions.shp'
    calc_crop_stats(cdl_dir, output_dir, None)

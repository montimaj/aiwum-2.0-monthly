# Author: Sayantan Majumdar
# Email: smxnv@mst.edu
# Data provided by Dr. Jordan Wilson, USGS
# MAP ML project under Missouri S&T and USGS MOU

# --------------------------------------- Running HydroMAP_ML -------------------------------------------------------

# Copy source files and saved CSV data to the HPC on Linux or Mac
# scp -r ./Python_Files/ "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/"
# scp -r ./Data/main/Annual_Subset.csv ./Data/main/VMP_readings_2014_2020.csv \
# "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/Data/main/"

# Copy source files and saved CSV data to the HPC on Windows powershell
# scp -r .\Python_Files\ "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/"
# scp -r .\Data\main\Annual_Subset.csv .\Data\main\VMP_readings_2014_2020.csv `
# "um-ad\smxnv@r07smithryang.managed.mst.edu:/usr/local/home/smxnv/HydroMAP_ML/Data/main/"

# Execute map_ml.py (change the paths and flags accordingly) on Linux or Mac
# python map_ml.py \
# --input-csv ../Data/main/VMP_readings_2014_2020.csv \
# --field-shp-dir ../USGS_MAP/permitted_boundaries/Shapefiles/ \
# --load-files True \
# --load-data True \
# --load-model True \
# --use-sub-cols True \
# --sub-cols ReportYear AF_Acre Latitude Longitude 'Crop(s)' \
# --test-size 0.2 \
# --random-state 1234 \
# --output-dir ../Outputs/ \
# --model-dir ../Models/ \
# --pred-attr AF_Acre \
# --start-month 4 \
# --end-month 9 \
# --data-list EEMETRIC PT-JPL SIMS MOD16 SSEBop SM_IDAHO SMOS_SMAP ppt tmin tmax tmean RO DEF SWB_HSG SWB_ET SWB_PPT \
# SWB_INT SWB_IRR SWB_INF SWB_RINF SWB_RO SWB_SS SWB_MRD SWB_SSM SWB_AWC \
# --gee-scale 1000 \
# --prism-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/PRISM/PRISM_800m/AN81/ \
# --cdl-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/CDL/ \
# --openet-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/OpenET/ \
# --eemetric-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/EEMETRIC/Monthly_ET_Rasters/ \
# --pt-jpl-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/PT-JPL/Monthly_ET_Rasters/ \
# --sims-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/SIMS/Monthly_ET_Rasters/ \
# --map-extent-file ../Data/Extent/MAP_Extent_New.shp \
# --year-col ReportYear \
# --crop-col 'Crop(s)' \
# --shuffle-split False \
# --train-year-list 2014 2015 2016 2017 2018 2019 2020 \
# --scaling True \
# --split-strategy 2 \
# --test-years 2020 \
# --load-map-csv False \
# --model-name BRF \
# --randomized-search True \
# --fold-count 5 \
# --repeats 3 \
# --drop-attr SMOS_SMAP DEF tmin tmean PT-JPL SIMS EEMETRIC MOD16 Relative_SSEBop Relative_PT-JPL `
# Relative_SIMS Relative_EEMETRIC Relative_MOD16 CC SM_IDAHO PermitNumb Crop_CDL `
# --crop-models False \
# --outlier-op 1 \
# --compare-aiwums False \
# --gdal-path /usr/bin/gdal/ \
# --gee-files RO \
# --prism-files ppt tmin tmax tmean \
# --load-pred-raster False \
# --load-pred-csv False \
# --load-map-extent False \
# --aiwum1-cdl-dir ../USGS_MAP/AIWUM1_1_output/high_res_AIWUM_output_bycrop_July/ \
# --aiwum1-tot-dir ../USGS_MAP/AIWUM1_1_output/Annual_Total/ \
# --comp-aiwum-verbose True \
# --pred-year-list 2014 2015 2016 2017 2018 2019 \
# --use-dask True

# Execute map_ml.py (change the paths and flags accordingly) on Windows powershell
# python map_ml.py `
# --input-csv ../Data/main/VMP_readings_2014_2020.csv `
# --field-shp-dir ../USGS_MAP/permitted_boundaries/Shapefiles/ `
# --load-files True `
# --load-data True `
# --load-model True `
# --use-sub-cols True `
# --sub-cols ReportYear AF_Acre Latitude Longitude 'Crop(s)' `
# --test-size 0.2 `
# --random-state 1234 `
# --output-dir ../Outputs/ `
# --model-dir ../Models/ `
# --pred-attr AF_Acre `
# --start-month 4 `
# --end-month 9 `
# --data-list EEMETRIC PT-JPL SIMS MOD16 SSEBop SM_IDAHO SMOS_SMAP ppt tmin tmax tmean RO DEF SWB_HSG SWB_ET SWB_PPT `
# SWB_INT SWB_IRR SWB_INF SWB_RINF SWB_RO SWB_SS SWB_MRD SWB_SSM SWB_AWC `
# --gee-scale 1000 `
# --prism-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/PRISM/PRISM_800m/AN81/ `
# --cdl-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/CDL/ `
# --openet-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/OpenET/ `
# --eemetric-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/EEMETRIC/Monthly_ET_Rasters/ `
# --pt-jpl-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/PT-JPL/Monthly_ET_Rasters/ `
# --sims-path ../USGS_MAP/MAP_project/files_from_Jordan/AIWUM_input_datasets/ET_Data/SIMS/Monthly_ET_Rasters/ `
# --map-extent-file ../Data/Extent/AIWUM1_Extent.shp `
# --year-col ReportYear `
# --crop-col 'Crop(s)' `
# --shuffle-split False `
# --train-year-list 2014 2015 2016 2017 2018 2019 2020 `
# --scaling True `
# --split-strategy 2 `
# --test-years 2020 `
# --load-map-csv True `
# --model-name BRF `
# --randomized-search True `
# --fold-count 5 `
# --repeats 3 `
# --drop-attr SMOS_SMAP DEF tmin tmean PT-JPL SIMS EEMETRIC Crop_CDL MOD16 Relative_SSEBop Relative_PT-JPL `
# Relative_MOD16 Relative_EEMETRIC PT-JPL CC SM_IDAHO PermitNumb Relative_SIMS `
# --crop-models False `
# --outlier-op 2 `
# --compare-aiwums True `
# --gdal-path C:/OSGeo4W64/ `
# --gee-files RO `
# --prism-files ppt tmin tmax tmean `
# --load-pred-raster False `
# --load-pred-csv False `
# --load-map-extent False `
# --aiwum1-cdl-dir ../USGS_MAP/AIWUM1_1_output/high_res_AIWUM_output_bycrop_July/ `
# --aiwum1-tot-dir ../USGS_MAP/AIWUM1_1_output/Annual_Total/ `
# --comp-aiwum-verbose False `
# --pred-year-list 2014 2015 2016 2017 2018 2019 `
# --use-dask False `
# --swb-data-path C:/Downloads/SWB_Data/

# ------------------------------------------------- Main code begins --------------------------------------------------


import argparse

from maplibs.dataops import prepare_data, create_prediction_map
from maplibs.dataops import compare_aiwums_map, create_train_test_data
from maplibs.mlops import build_ml_model, build_crop_ml_models, get_prediction_results, calc_train_test_metrics
from maplibs.sysops import boolean_string


def run_map_ml(args):
    """
    Driver function to prepare data and run ML models.
    :param args: Namespace containing the following keys:

    Keys: Description
    ___________________________________________________________________________________________________________________
    input_csv: Input MAP CSV file obtained from the USGS
    field_shp_dir: Field polygon shapefile directory path
    load_files: Set True to load existing data sets
    load_data: Set to True to load already downloaded data
    load_model: Set True to load pre-built model
    use_sub_cols: Set True to make subset data using sub_cols
    sub_cols: List of columns for creating subset data
    lat_pump: Name of the latitude column of the VMP data
    lon_pump: Name of the longitude column of the VMP data
    field_permit_col: Field permit column name
    test_size: Test data size
    random_state: PRNG seed
    output_dir: Output directory
    model_dir: Model directory
    pred_attr: Prediction/target attribute name
    start_month: Start month for downloading data sets
    end_month: End month for downloading data sets
    data_list: List of data sets to use/download. Valid names include 'SSEBop', 'SM_IDAHO', 'MOD16', 'SMOS_SMAP',
    'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', 'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin',
    'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', 'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF',
    'SWB_RINF', 'SWB_RO', 'SWB_SS', 'SWB_MRD', 'SWB_SSM', 'SWB_AWC'
    Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, 'CDL' for USDA-NASS cropland data,
    'SWB*' for SWB products.
    gee_scale: Google Earth Engine scale (m) for downloading
    prism_path: Path to the 800 m monthly PRISM products. Required if PRISM products are in data-list
    cdl_path: Path to the 30 m CDL products. Required if CDL is in data-list
    openet_path: Path to the annual OpenET products. Required if OpenET is in data-list
    eemetric_path: Path to the monthly EEMETRIC products. Required if EEMETRIC is in data-list
    pt_jpl_path: Path to the monthly PT-JPL products. Required if PT-JPL is in data-list
    sims_path: Path to the monthly SIMS products. Required if SIMS is in data-list
    map_extent_file: Path to the MAP extent shapefile for handling OpenET, EEMETRIC, PT-JPL, and SIMS.
    Required if any or all of OpenET, EEMETRIC, PT-JPL, and SIMS are in data-list
    year_col: Name of the year column
    crop_col: Name of the crop column
    shuffle_split: Set True to use ShuffleSplit instead of KFold
    train_year_list: Years to use in the model
    scaling: Whether to perform feature scaling (automatically False if crop_models is True)
    split_strategy: If 1, Split train test data based on year_col.
    If 2, then test_size amount of data from year_col or crop_col are kept for testing and rest for training;
    for this option, test-years should have some value other than None, else splitting is based on crop_col.
    For any other value of split-strategy, the data are randomly split.
    test_years: Years to purely keep as test data. split-attribute must be true to use this feature
    load_map_csv: Set True to use existing train and test data
    model_name: Set model name. Valid names include 'LGBM', 'BRF', 'RF', 'ETR', 'DT', 'BT', 'ABR', 'KNN', 'LR'
    randomized_search: Set False to use the exhaustive GridSearchCV
    fold_count: Number of folds for kFold
    repeats: Number of repeats for kFold
    drop_attr: Attributes to drop from the modeling process
    crop_models: Set True to build crop-specific ML models
    outlier_op: Outlier operation to perform. Set to 1 for removing outlier directly, 2 for removing outliers
    by each crop, or 3 for removing outliers by each year
    compare_aiwums: Set True to generate AIWUM 2.0 prediction maps and compare with AIWUM 1.1
    gdal_path: GDAL path. This is required if compare-aiwums is True
    gee_files: GEE data set names such as MOD16, RO, etc. This is required if compare-aiwums is True
    prism_files: PRISM data set names such as ppt, tmax, tmin, tmean. This is required if compare-aiwums is True
    load_pred_raster: Load existing prediction rasters, required if compare-aiwums is True
    load_pred_csv: Load existing prediction CSVs, required if compare-aiwums is True
    load_map_extent: Set True to load existing MAP extent rasters, required if compare-aiwums is True
    aiwum1_cdl_dir: AIWUM 1.1 CDL path, required if compare-aiwums is True
    aiwum1_tot_dir: AIWUM 1.1 annual prediction raster path, required if compare-aiwums is True
    comp_aiwum_verbose: Set True for extra info during AIWUM comparison
    pred_year_list: Years to predict
    use_dask: Flag for using dask
    swb_data_path: Path to the SWB data sets
    ___________________________________________________________________________________________________________________
    :return: None
    """

    if not args.use_sub_cols:
        args.sub_cols = []
    input_df, file_dirs = prepare_data(args.input_csv, args.field_shp_dir, args.sub_cols, data_list=args.data_list,
                                       data_start_month=args.start_month, data_end_month=args.end_month,
                                       already_prepared=args.load_files, skip_download=args.load_data,
                                       gee_scale=args.gee_scale, prism_data_path=args.prism_path,
                                       cdl_data_path=args.cdl_path, openet_data_path=args.openet_path,
                                       eemetric_data_path=args.eemetric_path, pt_jpl_data_path=args.pt_jpl_path,
                                       sims_data_path=args.sims_path, swb_data_path=args.swb_data_path,
                                       map_extent_file=args.map_extent_file, year_col=args.year_col,
                                       lat_pump=args.lat_pump, lon_pump=args.lon_pump,
                                       field_permit_col=args.field_permit_col)
    if args.crop_models:
        args.scaling = False
    ret_vals = create_train_test_data(input_df, args.output_dir, pred_attr=args.pred_attr, drop_attr=args.drop_attr,
                                      test_size=args.test_size, random_state=args.random_state, scaling=args.scaling,
                                      already_created=args.load_map_csv, year_col=args.year_col, crop_col=args.crop_col,
                                      year_list=args.train_year_list, split_strategy=args.split_strategy,
                                      test_year=args.test_years, outlier_op=args.outlier_op,
                                      crop_models=args.crop_models)
    x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test = ret_vals
    if not args.crop_models:
        model = build_ml_model(x_train, y_train, args.model_dir, args.model_name, args.random_state,
                               args.load_model, args.fold_count, args.repeats, y_scaler, args.randomized_search,
                               args.test_size, args.shuffle_split, args.use_dask)
        pred_df = get_prediction_results(model, x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train,
                                         year_test, args.model_dir, args.model_name, args.year_col, args.crop_col)
        calc_train_test_metrics(pred_df, args.crop_col, args.year_col)
    else:
        model = build_crop_ml_models(x_train, x_test, y_train, y_test, year_train, year_test, args.year_col,
                                     args.crop_col, args.model_dir, args.model_name, args.random_state, args.load_model,
                                     args.fold_count, args.repeats, x_scaler, y_scaler, args.randomized_search,
                                     args.test_size, args.shuffle_split, args.use_dask)
    if args.compare_aiwums:
        pred_data_list = [dl for dl in args.data_list if dl not in args.drop_attr]
        _, pred_wu_dir, map_extent_raster_dir = create_prediction_map(model, args.map_extent_file, file_dirs,
                                                                      args.output_dir, args.pred_year_list,
                                                                      pred_data_list, args.aiwum1_cdl_dir,
                                                                      args.start_month, args.end_month,
                                                                      args.crop_col, args.load_pred_csv,
                                                                      args.load_map_extent, args.load_pred_raster,
                                                                      args.comp_aiwum_verbose, args.gee_files,
                                                                      args.prism_files, x_scaler, y_scaler,
                                                                      args.gdal_path)
        compare_aiwums_map(args.aiwum1_tot_dir, pred_wu_dir, args.map_extent_file, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flags to run HydroMAP_ML')
    parser.add_argument('--input-csv', type=str, required=True, help='Input MAP VMP CSV file obtained from the USGS')
    parser.add_argument('--field-shp-dir', type=str, required=True, help='Field polygon shapefile directory path')
    parser.add_argument('--load-files', type=boolean_string, default=True,
                        help='Set True to load existing data sets')
    parser.add_argument('--load-data', type=boolean_string, default=True,
                        help='Set to True to load already downloaded data')
    parser.add_argument('--load-model', type=boolean_string, default=False,
                        help='Set True to load pre-built model')
    parser.add_argument('--use-sub-cols', type=boolean_string, default=False,
                        help='Set True to make subset data using sub-cols')
    parser.add_argument('--sub-cols', type=str, nargs='+', default=[],
                        help='List of columns for creating subset data')
    parser.add_argument('--lat-pump', type=str, default='Latitude', help='Name of the latitude column of the VMP data')
    parser.add_argument('--lon-pump', type=str, default='Longitude',
                        help='Name of the longitude column of the VMP data')
    parser.add_argument('--field-permit-col', type=str, default='PermitNumb', help='Field permit column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test data size')
    parser.add_argument('--random-state', type=int, default=43, help='PRNG seed')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model-dir', type=str, required=True, help='Model directory')
    parser.add_argument('--pred-attr', type=str, required=True, help='Prediction/target attribute name')
    parser.add_argument('--start-month', type=int, required=True, help='Start month for downloading data sets')
    parser.add_argument('--end-month', type=int, required=True, help='End month for downloading data sets')
    parser.add_argument('--data-list', type=str, nargs='+', required=True,
                        help="List of data sets to use/download. Valid names include 'SSEBop', 'SM_IDAHO', 'MOD16',"
                             " 'SMOS_SMAP', 'DROUGHT', 'PRISM', 'TMIN', 'TMAX', 'WS', 'RO', 'NDWI', 'SPH', 'DEF', "
                             "'VPD', 'VPD_SMAP', 'ppt', 'tmax', 'tmin', 'tmean', 'CDL', 'EEMETRIC', 'PT-JPL', 'SIMS', "
                             "'SWB_HSG', 'SWB_ET', 'SWB_PPT', 'SWB_INT', 'SWB_IRR', 'SWB_INF', 'SWB_RINF', 'SWB_RO', "
                             "'SWB_SS', 'SWB_MRD', 'SWB_SSM', 'SWB_AWC'"
                             "Note: 'ppt', 'tmax', 'tmin', 'tmean' are for PRISM 800 m data, "
                             "'CDL' for USDA-NASS cropland data, 'SWB*' for SWB products")
    parser.add_argument('--gee-scale', type=int, default=1000, help='Google Earth Engine scale (m) for downloading')
    parser.add_argument('--prism-path', type=str, default='',
                        help='Path to the 800 m monthly PRISM products. Required if PRISM products are in data-list')
    parser.add_argument('--cdl-path', type=str, default='',
                        help='Path to the 30 m CDL products. Required if CDL is in data-list')
    parser.add_argument('--openet-path', type=str, default='',
                        help='Path to the annual OpenET products. Required if OpenET is in data-list')
    parser.add_argument('--eemetric-path', type=str, default='',
                        help='Path to the monthly EEMETRIC products. Required if EEMETRIC is in data-list')
    parser.add_argument('--pt-jpl-path', type=str, default='',
                        help='Path to the monthly PT-JPL products. Required if PT-JPL is in data-list')
    parser.add_argument('--sims-path', type=str, default='',
                        help='Path to the monthly SIMS products. Required if SIMS is in data-list')
    parser.add_argument('--map-extent-file', type=str, default='',
                        help='Path to the MAP extent shapefile for handling OpenET, EEMETRIC, PT-JPL, and SIMS. '
                             'Required if any or all of OpenET, EEMETRIC, PT-JPL, and SIMS are in data-list')
    parser.add_argument('--year-col', type=str, required=True, help='Name of the year column')
    parser.add_argument('--crop-col', type=str, required=True, help='Name of the crop column')
    parser.add_argument('--shuffle-split', type=boolean_string, default=False,
                        help='Set True to use ShuffleSplit instead of KFold')
    parser.add_argument('--train-year-list', type=int, nargs='+', required=True, help='Years to train the model')
    parser.add_argument('--scaling', type=boolean_string, default=True, help='Whether to perform feature scaling')
    parser.add_argument('--split-strategy', type=int, default=3,
                        help='If 1, Split train test data based on a particular attribute like year_col or crop_col.'
                             'If 2, then test_size amount of data from year_col or crop_col are kept for testing and '
                             'rest for training; for this option, test-years should have some value '
                             'other than None, else splitting is based on crop_col. '
                             'For any other value of split-strategy, the data are randomly split.')
    parser.add_argument('--test-years', type=int, nargs='+', default=None,
                        help='Years to purely keep as test data. split-attribute must be true to use this feature')
    parser.add_argument('--load-map-csv', type=boolean_string, default=False,
                        help='Set True to use existing train and test data')
    parser.add_argument('--model-name', type=str, required=True,
                        help="Set model name. Valid names include 'LGBM', 'BRF', 'RF', 'ETR', 'DT', 'BT', 'ABR', "
                             "'KNN', 'LR'")
    parser.add_argument('--randomized-search', type=boolean_string, default=True,
                        help='Set False to use the exhaustive GridSearchCV')
    parser.add_argument('--fold-count', type=int, default=5, help='Number of folds for kFold')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats for kFold')
    parser.add_argument('--drop-attr', type=str, nargs='+', default=[],
                        help='Attributes to drop from the modeling process')
    parser.add_argument('--crop-models', type=boolean_string, default=False,
                        help='Set True to build crop-specific ML models')
    parser.add_argument('--outlier-op', type=int, default=None,
                        help='Outlier operation to perform. Set to 1 for removing outlier directly, '
                             '2 for removing outlier by each crop, or 3 for removing outliers by each year')
    parser.add_argument('--compare-aiwums', type=boolean_string, default=False,
                        help='Set True to generate AIWUM 2.0 prediction maps and compare with AIWUM 1.1')
    parser.add_argument('--gdal-path', type=str, help='GDAL path. This is required if compare-aiwums is True')
    parser.add_argument('--gee-files', type=str, nargs='+', default=[],
                        help='GEE data set names such as MOD16, RO, etc. This is required if compare-aiwums is True')
    parser.add_argument('--prism-files', type=str, nargs='+', default=[],
                        help='PRISM data set names such as ppt, tmax, tmin, tmean. '
                             'This is required if compare-aiwums is True')
    parser.add_argument('--load-pred-raster', type=boolean_string, default=False,
                        help='Load existing prediction rasters, required if compare-aiwums is True')
    parser.add_argument('--load-pred-csv', type=boolean_string, default=False,
                        help='Load existing prediction CSVs, required if compare-aiwums is True')
    parser.add_argument('--load-map-extent', type=boolean_string, default=False,
                        help='Set True to load existing MAP extent rasters, required if compare-aiwums is True')
    parser.add_argument('--aiwum1-cdl-dir', type=str, help='AIWUM 1.1 CDL path, required if compare-aiwums is True')
    parser.add_argument('--aiwum1-tot-dir', type=str,
                        help='AIWUM 1.1 annual prediction raster path, required if compare-aiwums is True')
    parser.add_argument('--comp-aiwum-verbose', type=boolean_string, default=False,
                        help='Set True for extra info during AIWUM comparison')
    parser.add_argument('--pred-year-list', type=int, nargs='+', required=True, help='Years to predict')
    parser.add_argument('--use-dask', type=boolean_string, default=True, help='Set False to disable Dask')
    parser.add_argument('--swb-data-path', type=str, required=True, help='Path to SWB data')

    map_args = parser.parse_args()
    run_map_ml(map_args)

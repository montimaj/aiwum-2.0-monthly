# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import geopandas as gpd
import rasterio as rio
import os
import pandas as pd
import gdal
from .rasterops import map_nodata, get_raster_extent, read_raster_as_arr


def reproject_vector(input_vector_file, outfile_path, ref_file, crs='epsg:4326', crs_from_file=True, raster=True):
    """
    Reproject a vector file
    :param input_vector_file: Input vector file path
    :param outfile_path: Output vector file path
    :param crs: Target CRS
    :param ref_file: Reference file (raster or vector) for obtaining target CRS
    :param crs_from_file: If true (default) read CRS from file (raster or vector)
    :param raster: If true (default) read CRS from raster else vector
    :return: Reprojected vector file in GeoPandas format
    """

    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    else:
        crs = {'init': crs}
    output_vector_file = input_vector_file.to_crs(crs)
    output_vector_file.to_file(outfile_path)
    return output_vector_file


def csv2shp(input_csv_file, outfile_path, delim=',', source_crs='epsg:4326', target_crs='epsg:4326',
            long_lat_field_names=('Longitude', 'Latitude')):
    """
    Convert CSV to Shapefile
    :param input_csv_file: Input CSV file path
    :param outfile_path: Output file path
    :param delim: CSV file delimiter
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param long_lat_field_names: Tuple containing names of the longitude and latitude columns respectively
    :return: None
    """

    input_df = pd.read_csv(input_csv_file, delimiter=delim)
    input_df = input_df.dropna(axis=1)
    long, lat = input_df[long_lat_field_names[0]], input_df[long_lat_field_names[1]]
    crs = {'init': source_crs}
    gdf = gpd.GeoDataFrame(input_df, crs=crs, geometry=gpd.points_from_xy(long, lat))
    gdf.to_file(outfile_path)
    if target_crs != source_crs:
        reproject_vector(outfile_path, outfile_path=outfile_path, crs=target_crs, crs_from_file=False, ref_file=None)


def shp2raster(input_shp_file, outfile_path, value_field=None, value_field_pos=0, xres=1000., yres=1000.,
               add_value=True, output_crs='EPSG:4326', ref_raster=None):
    """
    Convert Shapefile to Raster TIFF file using GDAL rasterize
    :param input_shp_file: Input Shapefile path
    :param outfile_path: Output TIFF file path
    :param value_field: Name of the value attribute. Set None to use value_field_pos
    :param value_field_pos: Value field position (zero indexing)
    :param xres: Pixel width in geographic units
    :param yres: Pixel height in geographic units
    :param add_value: Set False to disable adding value to existing raster cell
    :param output_crs: Output raster CRS
    :param ref_raster: Set to reference raster file path for creating the new raster as per this reference
    raster SRS and resolution
    :return: None
    """

    ext_pos = input_shp_file.rfind('.')
    sep_pos = input_shp_file.rfind(os.sep)
    if sep_pos == -1:
        sep_pos = input_shp_file.rfind('/')
    layer_name = input_shp_file[sep_pos + 1: ext_pos]
    shp_file = gpd.read_file(input_shp_file)
    if value_field is None:
        value_field = shp_file.columns[value_field_pos]
    if not ref_raster:
        minx, miny, maxx, maxy = shp_file.geometry.total_bounds
    else:
        _, ref_file = read_raster_as_arr(ref_raster)
        minx, miny, maxx, maxy = get_raster_extent(ref_file, is_rio_obj=True)
        xres, yres = ref_file.res
        output_crs = ref_file.crs.data['init']
    no_data_value = map_nodata()
    rasterize_options = gdal.RasterizeOptions(
        format='GTiff', outputType=gdal.GDT_Float32,
        outputSRS=output_crs,
        outputBounds=[minx, miny, maxx, maxy],
        xRes=xres, yRes=yres, noData=no_data_value,
        initValues=0., layers=[layer_name],
        add=add_value, attribute=value_field
    )
    gdal.Rasterize(outfile_path, input_shp_file, options=rasterize_options)

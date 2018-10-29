"""
import logging, os
import geopandas as gpd
import numpy as np
import rasterio as rio
from datetime import datetime
import Utilities as utils

logger = logging.getLogger(__name__)

def calculateGeom():
    print("OK")
def get_VegIndicies_VRT():
    print("OK")
def get_GaussianFile():
    print("OK")
def createSubSetLandsat():
    print("OK")
def get_values():
    print("OK")



def extractToPoints(training_points, out_file, data_dir, landsat_file, rasters_names, overright=False):
    ext_start = datetime.now()
    base_landsatdir = os.path.join(data_dir, "Landsat8")
    landsat_dir = os.path.join(base_landsatdir, "byNAIPDOY_QQuads")
    ndwi_dir = os.path.join(data_dir, "NDWI")
    ndsi_dir = os.path.join(data_dir, "NDSI")
    # LOCATION OF THE NDSI FILE
    ndsi_file = os.path.join(ndsi_dir, "LandsatOLI_NDSI_30m.tif")
    # LOCATION OF THE NDWI FILE
    ndwi_file = os.path.join(ndwi_dir, "LandsatOLI_NDWI_30m.tif")

    utils.useDirectory(landsat_dir)
    utils.useDirectory(ndwi_dir)
    utils.useDirectory(ndsi_dir)

    # IF VECTOR FILE OF POINTS WITH RASTER EXTRACTS DOESN'T EXIST, BUILD IT
    if not os.path.exists(out_file) or overright:
        #if "class_points" not in locals():
        #    logging.debug("READING IN %s as class_points" % training_points)
        training_data_df = gpd.read_file(training_points, crs={'init': 'epsg:26912'})

        if "utm_geom" not in training_data_df:
            logging.debug("ADDING COLUMN 'utm_geom' WITH CORRECT UTM COORDINATES FOR EACH QUARTER QUAD")
            # CREATE TRUE RASTER GEOMETRY COLUMN (BASED ON UTM)
            training_data_df["utm_geom"] = training_data_df.apply(calculateGeom, axis=1)

        # NDSI is only used because its the last raster column
        if "NDSI" not in training_data_df:
            logging.debug("CREATING COLUMNS...")
            # CREATE EMPTY COLUMNS IN DATA FRAME FOR EACH RASTER VARIABLE
            for column in rasters_names:
                training_data_df[column] = np.NaN

        net_percentage = 0.0
        # ITERATE THROUGH DATAFRAME IN GROUPS BY NAIP_FILE. KEEPS US FROM OPENING/CLOSING RASTERS FOR EACH POINT - INSTEAD FOR EACH GROUP
        for loc_NAIPFile, group in training_data_df.groupby("NAIP_FILE"):
            logger.debug("\nStarting raster value extraction for points in qquad %s" % loc_NAIPFile)
            loc_NAIPFile.replace("\\", "/")  # normalize for windows paths

            # LOOK FOR RASTERS FROM WHICH VALUES WILL BE EXTRACTED
            file = os.path.basename(loc_NAIPFile)

            vrt_naipvis = get_VegIndicies_VRT(file)
            # vrt_stddev = get_STDDev_VRT(file)
            gaussf_path = get_GaussianFile(file)

            landsat_path = createSubSetLandsat(loc_NAIPFile, landsat_file, landsat_dir).replace("\\", "/")

            landsat_ndsi_path = createSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_dir).replace("\\", "/")
            landsat_ndwi_path = createSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_dir).replace("\\", "/")

            net_percentage += 100 * len(training_data_df.loc[training_data_df["NAIP_FILE"] == loc_NAIPFile]) / len(training_data_df)
            logger.debug("Percentage of total: %d" % net_percentage)
            # SELECT POINTS WHICH HAVE NAIP PATH VALUE

            # Only if group hasn't had values assigned (Jupyter and Rodeo iterations)
            if group["NDSI"].isnull().values.any():
                with rio.open(loc_NAIPFile) as rasnaip:
                    with rio.open(vrt_naipvis) as rasnaipvis:
                        with rio.open(gaussf_path) as rasgauss:
                            with rio.open(landsat_path) as raslandsat:
                                with rio.open(landsat_ndsi_path) as rasNDSI:
                                    with rio.open(landsat_ndwi_path) as rasNDWI:
                                        count = 0
                                        training_data_df.loc[training_data_df.NAIP_FILE == loc_NAIPFile, rasters_names] = \
                                            training_data_df.loc[training_data_df.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(
                                                get_values)

            logger.debug("Finished with group %s at %s" % (loc_NAIPFile, str(datetime.now())))

        # REMOVE ALL ROWS WHICH EXTRACTED NO DATA VALUES FROM LANDSAT
        # for column in landsat:
        #    class_points = class_points[class_points.loc[column] != 32766]

        logger.info("Finished raster value extraction of %s points in %s" % (
        str(len(training_data_df)), str(datetime.now() - ext_start)))

        # GEOPANDAS WON"T ALLOW MORE THAN ONE COLUMN WITH GEOMETRY TYPE. REMOVE THE utm_geom COLUMN CREATED PREVIOUSLY
        del training_data_df['utm_geom']
        # print("COLUMNS:\n", training_data_df.columns)
        logger.debug("WRITING DATAFRAME TO OUTPUT...")
        training_data_df.to_file(out_file)

    else:
        #if "training_data_df" not in "locals":
        logger.info("Reading in point file %s" % out_file)
        training_data_df = gpd.read_file(out_file)
        # Had to delete utm_geom when writing file (can't have two geometry columns). Recreate...
        rasters_names = training_data_df.columns.tolist()[18:-1]

    return {"training_points": training_data_df, "raster_names": rasters_names}

"""
# Felzenswalb segmentation and mean

import logging
import rasterio as rio
import numpy as np
import os, shutil
import Utilities as utils
import numpy as np
import gdal
import geopandas as gpd
import glob
import fiona
import shutil
from Utilities import getFullNAIPPath
import logging
logger = logging.getLogger(__name__)

def mergeFlowlineSizeRasters(rasters, out_raster, cleanup):
    arrays = []
    for i in range(len(rasters)):
        with rio.open(rasters[i]) as ras:
            kwargs = ras.profile
            ras_array = ras.read().astype(float)
            arrays.append(ras_array)

    max_array = arrays[0]
    for i in range(len(arrays)-1):
        max_array = np.maximum(arrays[i], arrays[i+1])


    with rio.open(out_raster, 'w', **kwargs) as dst:
        dst.write(max_array.astype(rio.float32))


    """
    # Merge individual vb rasters to one
    logging.info("Merging raster sizes to final at %s" % flowlines_clipped_raster)
    with rio.open(clipped_rasters["Small"]) as small_ras:
        small_array = small_ras.read().astype(float)
    with rio.open(clipped_rasters["Medium"]) as med_ras:
        medium_array = med_ras.read().astype(float)
    with rio.open(clipped_rasters["Large"]) as large_ras:
        large_array = large_ras.read().astype(float)
        kwargs = large_ras.profile

    # numpy maximum only allows comparison of two arrays at a time. weird
    flowline_buffer_array = np.maximum(small_array, medium_array)
    flowline_buffer_array = np.maximum(flowline_buffer_array, large_array)

    with rio.open(flowlines_clipped_raster, 'w', **kwargs) as dst:
        dst.write(flowline_buffer_array.astype(rio.float32))

    if cleanup == True:
        for file in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, file)
            shutil.remove(fpath)
        os.rmdir(temp_dir)
    """

"""
training_naip = glob.glob(r"M:\Data\TrainingImageStack\*.tif")
alreadydone = glob.glob(r"M:\Data\TrainingImageStack\*.tif")
already_done = []
for ad in alreadydone:
    naip_name = os.path.basename(ad[:-18] + ".tif")
    #fullpath = getFullNAIPPath(naip_name, r"Q:\Arid Riparian Project\Data\\NAIP_2015_Compressed")
    already_done.append(naip_name)
print("ALREADY DONE:", len(already_done))
print(already_done)

base_datadir = os.path.abspath(r"M:\Data")
training_data_dir = os.path.join(base_datadir, "inital_model_inputs")
loc_usgs_qquads = os.path.join(training_data_dir, "USGS_QQuads_AZ.shp")
footprints = gpd.read_file(loc_usgs_qquads)
aoi = gpd.GeoDataFrame.from_file(r"Q:\Arid Riparian Project\AridRiparianProject\AridRiparianProject.gdb", layer='TargetEcoregions')

aoi.crs = fiona.crs.from_epsg(2163)
#print(aoi.crs)

footprints = gpd.read_file(loc_usgs_qquads)
aoi.to_crs(footprints.crs, inplace=True)

aoi_qquads = []
for i, row in footprints.iterrows():
    for j, arow in aoi.iterrows():
        if row.geometry.within(arow.geometry):
            fullpath = getFullNAIPPath(row.QUADID, r"Q:\Arid Riparian Project\Data\\NAIP_2015_Compressed")
            file = os.path.basename(fullpath)
            aoi_qquads.append(file)

print("AOI QQUADS: {}".format(len(aoi_qquads)))
print(aoi_qquads)

shared_calculation_qquads = aoi_qquads[:]

for quad in aoi_qquads:
    #for tf in already_done:
        #print(quad, tf)
     if quad in already_done:
        #print(quad)
        #fullpath = getFullNAIPPath(quad, r"Q:\Arid Riparian Project\Data\\NAIP_2015_Compressed")
        #print(fullpath, file=tfile)
        shared_calculation_qquads.remove(quad)

#print("TRAINING QUADS", list_of_training_qquads)



total_num = len(shared_calculation_qquads)
segment_size = int(total_num/4)
print("segment_size", segment_size)

for i in range(1,4+1):
    start_index = (i-1) * segment_size
    end_index = i * segment_size
    subset = shared_calculation_qquads[start_index : end_index]
    with open(r"M:\\NaipDone" + str(i) + ".txt", 'w') as tfile:
        for j in subset:
            print(j, file=tfile)
"""

"""
outdir = r"D:\Data\\NAIP"
in_txt_file = r"D:\Data\\NaipDone4.txt"
files = glob.glob(outdir+"\\*.tif")

sub_files = []
with open(in_txt_file, 'r') as txt:
    for l in txt:
        sub_files.append(l[:-1])

for fs in files:
    if fs not in sub_files:
        os.remove(os.path.join(outdir, fs))

for tf in sub_files:
    if not os.path.exists(os.path.join(outdir,tf)):
        fpath = getFullNAIPPath(tf, r"Q:\Arid Riparian Project\Data\\NAIP_2015_Compressed")
        shutil.copy(fpath, outdir)
"""


l = [1,2,3,4,5,6,7,8,9,0]

file = "RiparianClassification_3111001_nw.tif"
print(file[-14:-4])
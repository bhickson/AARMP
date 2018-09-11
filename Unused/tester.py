import pandas as pd
import glob, os, math
import rasterio as rio
from datetime import datetime
import shutil
import sys
import Utilities as utilities
import numpy as np
import gdal, osr
import geopandas as gpd


def findFile(dir, f):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if f in file:
                fpath = os.path.join(root, file)
                return fpath





"""
for root, dirs, files in os.walk(naip_dir):
    for file in files:
        #if file.endswith(".tif"):
        if file == naip_file:
            naip_path = os.path.join(root, file)

            print("Searching for Landsat file")
            landsat_path = findFile(landsat_dir, file)
            if landsat_path is None:

                reference_f = gdal.Open(naip_path)
                geo_transform = reference_f.GetGeoTransform()
                resx = geo_transform[1]
                resy = geo_transform[5]
                proj = reference_f.GetProjectionRef()
                minx = geo_transform[0]
                maxy = geo_transform[3]
                maxx = minx + (resx * reference_f.RasterXSize)
                miny = maxy + (resy * reference_f.RasterYSize)


                #build landsat tile from naip extent

                ofile = "Landsat8_" + file

                landsat_opath = os.path.join(landsat_dir, ofile)
                gdal_warp = "gdalwarp -overwrite -tap -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_merge, landsat_opath)
                print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_merge, naip_path))
                os.system(gdal_warp)

count = 0
"""
#dir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"
"""
rasters = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".tif"):
            fpath = os.path.join(root,file)
            rasters.append(fpath)

all_files = " ".join(rasters)

outputfile = os.path.join(dir, "33109.vrt")
gdal_merge_dir = "python gdal_merge.py  -o %s %s" % (outputfile, all_files)

"""
from datetime import datetime
import struct
import timeit

start = datetime.now()

#print("Starting GDAL Translate at", start)

#rasterDir = r"Q:\Arid Riparian Project\Data\az_1m_2015"
#outDir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"
#count = 0

def getProj(file):
    raster = gdal.Open(file)
    return raster.GetProjectionRef()

def usingRIO():
    with rio.open(r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed\m_3110913_sw_12_1_20150621.tif") as src:
        x = (src.bounds.left + src.bounds.right)/2.0
        y = (src.bounds.bottom + src.bounds.top)/2.0
        src.index(x, y)
        (359, 396)
        def tiny_window(dataset, x, y):
            r, c = dataset.index(x, y)
            return ((r, r+1), (c, c+1))

        data = src.read(window=tiny_window(src, x, y))
        print(list(data[:,0,0]))

#print(timeit.repeat("for x in range(10): usingRIO()", "from __main__ import usingRIO", number=10))
"""
ref_proj = getProj(os.path.join(outDir, "m_3310901_nw_12_1_20150614.tif")) # EPSG 26912

for root, dirs, files in os.walk(rasterDir):
    for file in files:
        if file.endswith(".tif"):
            fpath = os.path.join(root, file)
            ds = gdal.Open(fpath, gdal.GA_ReadOnly)
            proj = ds.GetProjectionRef()
            if proj != ref_proj:
                count += 1
                opath = os.path.join(outDir, file)
                geo_transform = ds.GetGeoTransform()
                resx = geo_transform[1]
                resy = geo_transform[5]
                minx = geo_transform[0]
                maxy = geo_transform[3]
                width = resx * ds.RasterXSize
                height = resy * ds.RasterYSize
                print("%d Executing gdal_translate on %s..." % (count, file))
                gdal_translate = "gdalwarp -overwrite -s_srs %s -t_srs %s -tr %s %s -tr %s %s %s %s" % (proj, ref_proj, resx, resy, width, height, fpath, opath)
                #print(file)
                os.system(gdal_translate)
            ds = None
print(count)"""


def convertDataType(indir, outdir):

    for file in os.listdir(indir):
        if file.endswith(".tif"):
            start = datetime.now()
            print("Starting %s" % file)
            fpath = os.path.join(indir, file)
            opath = os.path.join(outdir,file)
            with rio.open(fpath) as raster:
                kwargs = raster.profile

                ras = raster.read().astype(rio.uint16)

                kwargs.update(
                    dtype=rio.uint16,
                    crs = "+proj=utm +zone=12 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
                )

                with rio.open(opath, 'w', **kwargs) as dst:
                    dst.write(ras.astype(rio.uint16))
            print("\tFinished %s in %s" % (file, str(datetime.now() - start)))

dir = r"M:\Data\Landsat8\acqui_clips\uint16"

def mergeRasters(indir, outfile):
    import glob

    files = glob.glob(indir + "\\" + "*.tif")
    print("getting kwargs")
    with rio.open(files[0]) as ras:
        print("getting kwargs")
        kwargs = ras.profile

    vrt_file = os.path.join(indir, "day_200_index.vrt")

    print("Opening VRT...")
    with rio.open(vrt_file) as ras:
        raster = ras.read()

        print("Writing outfile...")
        with rio.open(outfile, 'w', **kwargs) as dst:
            dst.write(raster.astype(rio.uint16))

    print("FINISHED")

#mergeRasters(dir, os.path.join(dir, "landsat8.tif")

def createSubSetLandsat(): #naip_path, landsat_vrt):
    landsat_vrt = os.path.normpath(r"M:\Data\Landsat8\acqui_clips\landsat8_index.vrt")
    landsat_dir = r"M:\Data\Landsat8\byNAIPDOY"
    naip_dir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"

    for root, dirs, files in os.walk(naip_dir):
        for file in files:
            if file.endswith(".tif"):
                naip_path = os.path.join(root, file)
                print("\nStarting on %s" % naip_path)
                ofile = "Landsat8_" + os.path.basename(naip_path)
                landsat_opath = os.path.join(landsat_dir, ofile)

                if not os.path.exists(landsat_opath):
                    start = datetime.now()
                    reference_f = gdal.Open(naip_path)
                    geo_transform = reference_f.GetGeoTransform()
                    resx = geo_transform[1]
                    resy = geo_transform[5]
                    proj = reference_f.GetProjectionRef()
                    minx = geo_transform[0]
                    maxy = geo_transform[3]
                    maxx = minx + (resx * reference_f.RasterXSize)
                    miny = maxy + (resy * reference_f.RasterYSize)

                    # build landsat tile from naip extent

                    gdal_warp = "gdalwarp -overwrite -tap -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
                        proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_vrt, landsat_opath)
                    print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_vrt, naip_path))
                    os.system(gdal_warp)

                    print("FINISHED", datetime.now() - start)
                else:
                    print("File exists. Skipping %s" % landsat_opath)

def testReturn():
    #mset = tuple()
    values = [1,2,3]
    return tuple(values)

#print(testReturn())
"""
loc_class_points = r"Q:/Arid Riparian Project/Data/classificationPoints_join_26912.gpkg"
class_points = gpd.read_file(loc_class_points)

relevant_naip = class_points.NAIP_FILE.unique()
"""

#for file in relevant_naip:
#    createSubSetLandsat(r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed\m_3110937_nw_12_1_20150621.tif", r"M:\Data\Landsat8\acqui_clips\uint16\landsat8_index.vrt")
# runs in about 50 seconds per naip tile



def getValues():
    rasters = [r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed\m_3110901_ne_12_1_20150621.tif"]
    values = []

    x = 602485
    y = 3537208

    for raster in rasters:
        print("Starting Raster Extract for %s at x:%s y:%s" % (os.path.basename(raster), str(x), str(y)))
        with rio.open(raster) as ras:
            for val in ras.sample([(x, y)]):
                values += np.ndarray.tolist(val)

    print(values)
    print(pd.Series(values))

#getValues()




def createSubSetLandsat(naip_path, landsat_vrt, opath):
    ofile = "Landsat8_" + os.path.basename(naip_path)

    landsat_opath = os.path.join(opath, ofile)

    if os.path.exists(landsat_opath):
        return landsat_opath
    else:
        start = datetime.now()
        reference_f = gdal.Open(naip_path)
        geo_transform = reference_f.GetGeoTransform()
        resx = geo_transform[1]
        resy = geo_transform[5]
        proj = reference_f.GetProjectionRef()
        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + (resx * reference_f.RasterXSize)
        miny = maxy + (resy * reference_f.RasterYSize)

        # build landsat tile from naip extent


        gdal_warp = "gdalwarp -overwrite -tap -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
            proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_vrt, landsat_opath)
        print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_vrt, naip_path))
        os.system(gdal_warp)

        print("FINISHED", datetime.now() - start)

    return landsat_opath
"""
for file in os.listdir(indir):
    if file.endswith(".tif"):
        start = datetime.now()
        print("Starting %s" % file)
        fpath = os.path.join(indir, file)
        opath = os.path.join(outdir, file)
        with rio.open(fpath) as raster:
            kwargs = raster.profile

            ras = raster.read().astype(rio.uint16)

            kwargs.update(
                dtype=rio.uint16,
                crs="+proj=utm +zone=12 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
            )

            with rio.open(opath, 'w', **kwargs) as dst:
                dst.write(ras.astype(rio.uint16))
        print("\tFinished %s in %s" % (file, str(datetime.now() - start)))
"""

import numpy as np
import rasterio as rio
import os
from scipy import ndimage as ndi


def gaussianCalc(naip_file, oloc, sigma=1, overwrite=False):
    raster_paths = []
    # GET FILE NAME FROM PATH
    fname = os.path.basename(naip_file)
    sigma_dir = os.path.join(oloc, "Gauss_" + str(sigma))
    utilities.useDirectory(sigma_dir)

    out_file = os.path.join(sigma_dir, "gauss_" + str(sigma) + "_" + fname[:-4] + ".tif")

    if not overwrite and os.path.exists(out_file):
        print("SKIPPING %s..." % out_file)
    else:
        with rio.open(naip_file) as naip_ras:
            kwargs = naip_file.profile
            naip_array = naip_ras.read().astype(rio.float32)

        kwargs.update(
            dtype=rio.float32
        )

        gaussian_array = ndi.gaussian_filter(naip_array, 1)

        with rio.open(out_file, 'w', **kwargs) as dst:
            dst.write(gaussian_array.astype(rio.float32))

    return out_file


######################################################

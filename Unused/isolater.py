import logging, math
import geopandas as gpd
import rasterio as rio
import numpy as np
import fiona
from datetime import datetime
import os
import shutil
import pandas as pd
import gdal
from pyproj import transform, Proj

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.externals import joblib
from shapely.geometry import Point

import Utilities as utils
import Create_Classification_Points
from RasterCalculations import *
from StackGeneration import generateStack

import logging
from joblib import Parallel, delayed


def findVBRaster(qquad, naip_data_dir, overwrite=False):
    vb_start = datetime.now()
    logging.debug("Starting creation of subset of valley bottom...")
    naip_path = getFullNAIPPath(qquad, naip_data_dir)
    ofile = "ValleyBottom_" + qquad + ".tif"

    o_path = os.path.join(loc_valleybottoms, ofile)

    reference_f = gdal.Open(VBET_VB_loc)
    geo_transform = reference_f.GetGeoTransform()
    sproj = reference_f.GetProjectionRef()

    # TODO - Duplicative scripting. Exisits twice in this file and also in the VBET classification
    if not os.path.exists(o_path) or overwrite:
        reference_f = gdal.Open(naip_path)
        geo_transform = reference_f.GetGeoTransform()
        resx = geo_transform[1]
        resy = geo_transform[5]
        tproj = reference_f.GetProjectionRef()
        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + (resx * reference_f.RasterXSize)
        miny = maxy + (resy * reference_f.RasterYSize)

        resampletype = "bilinear"

        gdal_warp = "gdalwarp -overwrite -tap -r %s -s_srs %s -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
            resampletype, sproj, tproj, resx, resy, tproj, str(minx), str(miny), str(maxx), str(maxy), VBET_VB_loc,
            o_path)
        # print("Executing gdal_warp operation on %s for footprint of naip file %s" % (o_path, naip_path))
        os.system(gdal_warp)

        logging.debug("\tFinished VB subset in %s" % (str(datetime.now() - vb_start)))

    return o_path
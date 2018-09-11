""" Script matches a polygon feature class of USGS Quarter Quads to the file in the 
naip directory and adds a column to the Quarter Quads indicating the projection"""


try:
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import gdal
    import geopandas as gpd
    import rtree
    import osr
    import ogr
except:
    print("Unable to import. Exiting")
    exit()

import os
from datetime import datetime

import logging

from shapely.geometry import shape, Point, mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


start = datetime.now()

naip_dir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"

def get_projection(file):
    ras = gdal.Open(file)
    prj = ras.GetProjection()
    ras = None

    srs=osr.SpatialReference(wkt=prj)
    if srs.IsProjected:
        return srs.GetAttrValue('projcs')
    return srs.GetAttrValue('geogcs')


def findFile(quad_id):
    for root, dirs, files in os.walk(naip_dir):
        for file in files:
            if quad_id in file:
                fpath = os.path.join(root, file)
                return fpath


def set_projection(quadid):
    naip_file = findFile(quadid)
    proj = get_projection(naip_file)

    return pd.Series([proj,naip_file])


def getWKTfromShapefile(file):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(file)

    # from Layer
    layer = dataset.GetLayer()
    srs = layer.GetSpatialRef()
    return srs

loc_mergedPointsGPKG = "Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\ClassificationShapefiles_20180428\classificationPoints.gpkg"
loc_usgs_qquads = r"Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\USGS_QQuads_AZ.shp"


mergedPointsGPKG = gpd.read_file(loc_mergedPointsGPKG)
usgs_azqquads = gpd.read_file(loc_usgs_qquads)

print("Reprojecting quarter quads...")
usgs_azqquads_26912 = usgs_azqquads.to_crs(mergedPointsGPKG.crs, inplace=True)


print("Starting file and projection association to quarter quad data frame...")
usgs_azqquads_26912[["PROJ","NAIP_FILE"]] = usgs_azqquads_26912['QUADID'].apply(set_projection)

out_quad = os.path.splitext(loc_usgs_qquads)[0] + "_26912.shp"
if not os.path.exists(out_quad):
    print("Writing reprojected dataframe to file here...\n\t %s" % out_quad)
    usgs_azqquads_26912.to_file(driver="ESRI Shapefile", filename=out_quad)

#reprojected_qquads = os.path.splitext(loc_usgs_qquads)[0] + "_26912.shp"
#usgs_azqquads_26912 = gpd.read_file(out_quad)
# should already be aware of that they're in the same projection, but make sure
#usgs_azqquads_26912.crs = mergedPointsGPKG.crs

print("Starting point to quarter quad join...")
pointInPoly = gpd.sjoin(mergedPointsGPKG, usgs_azqquads_26912, op='within')

outjoin = os.path.splitext(loc_mergedPointsGPKG)[0] + "_join.shp"

print("Writing out classification points with joined quarter quad info to new shapefile here...\n\t %s" % outjoin)
pointInPoly.to_file(outjoin, driver="ESRI Shapefile")

print("Finished. ", datetime.now()-start, " elapsed")
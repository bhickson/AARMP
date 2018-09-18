# Script to take polygons of training data and convert to points by location of pixel centroid per a single naip tile

import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib


import numpy
import fiona
import rasterio
import rasterio.features
from affine import Affine
from shapely.geometry import shape, Point, mapping
import os
from datetime import datetime
import osr, gdal, ogr
from pyproj import transform, Proj



def getSnappedPixelLocation(geom_x, geom_y, ras_attribs, props):
    #print("X: ", geom_x, "Y: ", geom_y)
    #print("Ras_aff.c {}, Ras_aff.f {}".format(ras_attribs["ulx"], ras_attribs["uly"]))
    """ Returns set of upper-right snapped pixel locations in set as (x, y)"""
    pix_xsize = ras_attribs["res_x"]
    pix_ysize = ras_attribs["res_y"]

    # get pixel coordinates of the geometry's bounding box
    xvals = sorted([geom_x, ras_attribs["ulx"]])
    yvals = sorted([geom_y, ras_attribs["uly"]])

    diffx = xvals[1] - xvals[0]
    diffy = yvals[1] - yvals[0]
    #print("DIFFS: ", diffx, diffy)

    pixel_xdiff = float("{0:.11f}".format( diffx % pix_xsize ))  # get modulo pixel difference to float precision of 11 decimals
    pixel_ydiff = float("{0:.11f}".format( diffy % pix_ysize ))  # get modulo pixel difference to float precision of 11 decimals


    #print("PIXEL DIFF: ", pixel_xdiff, pixel_ydiff)
    #ul = geometry.bounds[0:2]  # lower left
    #lr = geometry.bounds[2:4]  # upper right

    #snapped pixel locations
    if pixel_xdiff > pix_xsize / 2:
        #print("YEP")
        snapped_ulx = geom_x + (pix_xsize - pixel_xdiff)
    else:
        snapped_ulx = geom_x - pixel_xdiff

    if abs(pixel_ydiff) > abs(pix_ysize / 2):
        snapped_uly = geom_y + (abs(pix_ysize) + pixel_ydiff)
    else:
        snapped_uly = geom_y - abs(pixel_ydiff)

    #print("SNAPPED", snapped_ulx, snapped_uly)
    if snapped_ulx % pix_xsize != ras_attribs["ulx"] % pix_xsize:
        print(snapped_ulx % pix_xsize)
        #print("Props:", props)
        raise ValueError("BAD PIXEL VALUE FOR ULX - ", snapped_ulx)

    if snapped_uly % pix_ysize != ras_attribs["uly"] % pix_ysize:
        print(snapped_uly % pix_ysize)
        raise ValueError("BAD PIXEL VALUE FOR ULY - ", snapped_uly)

    return {"x": snapped_ulx, "y": snapped_uly}


def getRasterTransform(rasterloc, t_epsg="26912"):
    ras_attr = {}
    with rasterio.open(rasterloc, 'r') as raster:
        t = raster.transform
        ras_attr["res_x"] = t.a
        ras_attr["res_y"] = t.e
        ras_attr["ulx"] = t.c
        ras_attr["uly"] = t.f

        #print(t)

        raster_epsg = raster.crs['init'].split(":")[1]
        if raster_epsg != t_epsg:
            #utm11 = Proj(init="epsg:26911")
            t_epsg = "epsg:" + t_epsg
            t_srs = Proj(init=t_epsg)
            s_epsg = "epsg:"+raster_epsg
            s_srs = Proj(init=s_epsg)
            point = Point(transform(s_srs, t_srs, ras_attr["ulx"], ras_attr["uly"]))

            ras_attr["ulx"] = point.x
            ras_attr["uly"] = point.y

    return ras_attr


def createShapefilePoints(polys, overwrite=False):

    point_schema = {'geometry': 'Point',
                    'properties': {'Type': 'str',
                                   'Class': 'int',
                                   'PROJ': 'str',
                                   'NAIP_FILE': 'str'}}

    parentdir = os.path.abspath(os.path.join(polys, os.pardir))
    out_point_file = os.path.join(parentdir, "classificationPoints.shp")
    start = datetime.now()

    if not os.path.exists(out_point_file) or overwrite:
        with fiona.open(polys, 'r') as vector:
            print("NUM POLY FEATURES: ", len(vector))

            poly_epsg = vector.crs['init'].split(":")[1]  # {'init': 'epsg:26912'}

            count = 0

            with fiona.open(out_point_file, 'w', crs=vector.crs, driver="ESRI Shapefile", schema=point_schema) as pointput:

                for feature in vector:
                    #print("Starting feature ", feature["id"])
                    geometry = shape(feature['geometry'])

                    naip_file = feature['properties']['NAIP_FILE']

                    # REFERENCE RASTER TO SNAP PIXELS TO
                    ras_atts = getRasterTransform(naip_file, poly_epsg)

                    geom_b = geometry.bounds  # eg (407923.6815999998, 3723410.0965, 407924.52249999996, 3723412.8594000004)

                    ul = getSnappedPixelLocation(geom_b[0], geom_b[3], ras_atts, feature['properties'])
                    lr = getSnappedPixelLocation(geom_b[2], geom_b[1], ras_atts, feature['properties'])
                    #print("UL:", ul)
                    #print("\tLR:", lr)

                    outshape_x = int(abs(lr["x"] - ul["x"]))
                    outshape_y = int(abs(ul["y"] - lr["y"]))
                    outshapex_inPixels = int(outshape_x / abs(ras_atts["res_x"]))
                    outshapey_inPixels = int(outshape_y / abs(ras_atts["res_y"]))

                    #if outshapex_inPixels == 0 or outshape_y == 0:
                        #print("geom: {}".format(geom_b))
                        #print(polygon_external_points)
                        #raise ValueError("Snapped bounding box is not correct", outshapex_inPixels, outshapey_inPixels)

                    half_x_size = abs(ras_atts["res_x"]) / 2
                    half_y_size = abs(ras_atts["res_y"]) / 2

                    polygon_internal_points = []
                    polygon_external_points = []

                    for x in range(outshapex_inPixels):
                        pointx = (ul["x"] + half_x_size) + (x * abs(ras_atts["res_x"]))
                        for y in range(outshapey_inPixels):
                            pointy = (ul["y"] - half_y_size) - (y * abs(ras_atts["res_y"]))
                            point = Point(pointx, pointy)

                            if point.within(geometry):
                                props = {'Type': str(feature["properties"]["Type"]),
                                         'Class': int(feature["properties"]['Class']),
                                         'PROJ': str(feature["properties"]["PROJ"]),
                                         'NAIP_FILE': str(feature["properties"]["NAIP_FILE"])}
                                polygon_internal_points.append(mapping(point))
                                pointput.write({'geometry': mapping(point), 'properties': props})
                                count += 1
                            else:
                                polygon_external_points.append(mapping(point))

                    if len(polygon_internal_points) == 0:
                        print("geom: {}".format(geom_b))
                        print(polygon_external_points)
                        #raise ValueError("PROBLEM. NO POINTS CREATED FOR FEATURE - ", feature)
                        print("PROBLEM. NO POINTS CREATED FOR FEATURE - ", feature)

            print("Finished", polys, " in ", datetime.now() - start, "\n\tCreated ", count, " points")

    return out_point_file


def get_projection(file):
    ras = gdal.Open(file)
    prj = ras.GetProjection()
    ras = None

    srs = osr.SpatialReference(wkt=prj)
    if srs.IsProjected:
        return srs.GetAttrValue('projcs')
    return srs.GetAttrValue('geogcs')


def findFile(quad_id, naip_dir):
    for root, dirs, files in os.walk(naip_dir):
        for file in files:
            if quad_id in file:
                fpath = os.path.join(root, file)
                return fpath


def getWKTfromShapefile(file):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(file)

    # from Layer
    layer = dataset.GetLayer()
    srs = layer.GetSpatialRef()
    return srs


def joinProjectionAndNAIP(classificationPolys_file, naip_directory, qquad_file, overwrite=False):
    """ Take a shapefile of classification polygons and finds the naip file covering
    the same area as the polygon as well as the projection"""

    def set_projection(quadid):
        naip_file = findFile(quadid, naip_directory)
        proj = get_projection(naip_file)

        return pd.Series([proj, naip_file])

    loc_outjoin = os.path.splitext(classificationPolys_file)[0] + "_join.shp"
    if not os.path.exists(loc_outjoin) or overwrite:
        classification_polys_df = gpd.read_file(classificationPolys_file)

        usgs_azqquads = gpd.read_file(qquad_file)

        print("Reprojecting quarter quads...")
        usgs_azqquads_26912 = usgs_azqquads.copy()

        usgs_azqquads_26912.to_crs(classification_polys_df.crs, inplace=True)

        print("Starting file and projection association to quarter quad data frame...")
        usgs_azqquads_26912[["PROJ", "NAIP_FILE"]] = usgs_azqquads_26912['QUADID'].apply(set_projection)

        print("Starting point to quarter quad join...")
        polysinQQuads = gpd.sjoin(classification_polys_df, usgs_azqquads_26912, op='within')

        columns_to_keep = ['geometry', 'Type', 'Class', 'PROJ', 'NAIP_FILE']
        drop_columns = polysinQQuads.columns.tolist()
        for c in columns_to_keep:
            drop_columns.remove(c)

        polysinQQuads.drop(drop_columns, axis=1, inplace=True)



        print(
            "Writing out classification points with joined quarter quad info to new shapefile here...\n\t %s" % loc_outjoin)
        polysinQQuads.to_file(loc_outjoin, driver="ESRI Shapefile")
        # --------------------------------------

    return loc_outjoin

    # polygonShapes_loc = r"Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\ClassificationShapefiles_20180428\PolygonShapefiles"


    # outdir = os.path.join(parentdir, "PointsShapefiles")
    # if not os.path.exists(outdir):
    #    os.mkdir(outdir)

    # mergedGPKG = os.path.join(parentdir, "classificationPoints.gpkg")  # location of geopackage which will hold all points


    """
    for file in os.listdir(polygonShapes_loc):
        if file.endswith(".shp"):
            inpath = os.path.join(polygonShapes_loc, file)
            outfilename = os.path.splitext(file)[0] + "_points.shp"

            outpath = os.path.join(outdir, outfilename)

            if not os.path.exists(outpath):
                createShapefilePoints(inpath, outpath, point_schema)


            oln = "points"  # out layer name
            if not os.path.exists(mergedGPKG):
                # CREATE NEW GEOPACKAGE FROM THE POINTS SHAPEFILE
                ogr_merge = "ogr2ogr -f GPKG %s %s -nln %s" % (mergedGPKG, outpath, oln)
            else:
                # APPEND THE POINTS SHAPEFILE DATA TO THE GEOPACKAGE
                ogr_merge = "ogr2ogr -f GPKG -update -append %s %s -nln %s" % (mergedGPKG, outpath, oln)

            #print("EXECUTING: ", ogr_merge)
            print("Merging %s into %s ...." % (outfilename, mergedGPKG))

            os.system(ogr_merge)"""

    # gpkg_gpd = gpd.read_file(mergedGPKG)


    # pointInPoly = gpd.sjoin(points, polys, op='within')



if __name__ == '__main__':
    print()
    naip_dir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"
    loc_classificationPolys = "M:\Data\inital_model_inputs\classificationTrainingPolygons.shp"
    loc_usgs_qquads = r"M:\Data\inital_model_inputs\USGS_QQuads_AZ.shp"

    loc_out_join = joinProjectionAndNAIP(loc_classificationPolys, naip_dir, loc_usgs_qquads, overwrite=False)

    loc_points_in_poly = createShapefilePoints(loc_out_join, overwrite=False)

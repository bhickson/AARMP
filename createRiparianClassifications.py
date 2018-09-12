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


def getFullNAIPPath(naip_file, naipdir):
    for root, dirs, files in os.walk(naipdir):
        for file in files:
            if naip_file in file:
                return os.path.join(root, file)

    logging.error("Unable to find naip file %s in %s. Exiting" % (naip_file, naipdir))
    raise Exception


def findSTDDevFile(dir, naip_file, band_num, windowsize):
    # findFile(os.path.join(std3px_dir, bandnum), ffile)

    window_dir = os.path.join(dir, "StdDev_" + str(windowsize) + "px")
    utilities.useDirectory(window_dir)
    band_dir = os.path.join(window_dir, "band" + band_num)
    utilities.useDirectory(band_dir)

    fname = os.path.basename(naip_file)

    for root, dirs, files in os.walk(band_dir):
        for file in files:
            if fname in file:
                fpath = os.path.join(root, file)
                return fpath

    if "fpath" not in locals():
        standardDeviation(naip_file, dir, window_size=windowsize, overwrite=False)

    logging.error("Unable to find standard deviation file for %s" % naip_file)
    raise Exception


def findVIFile(type, dir, f):
    fname = os.path.basename(f)
    for root, dirs, files in os.walk(dir):
        for file in files:
            if fname in file:
                fpath = os.path.join(root, file)
                return fpath

    if "fpath" not in locals():
        vegIndexCalc(f, dir, [type])
    return None


def createSubSetLandsat(naip_path, landsat_file, opath, overwrite=False):
    ssl_start = datetime.now()
    ofile = "Landsat8_" + os.path.basename(naip_path)

    landsat_opath = os.path.join(opath, ofile)

    if not os.path.exists(landsat_opath) or overwrite:
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

        if "ndsi" in opath.lower() or "ndwi" in opath.lower():
            resampletype = "bilinear"
        else:
            resampletype = "bilinear"
            # resampletype = "near"

        gdal_warp = "gdalwarp -overwrite -tap -r %s -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
            resampletype, proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_file,
            landsat_opath)
        logging.debug("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_file, naip_path))
        os.system(gdal_warp)

        logging.debug("\tFinished qquad for %s landsat in %s" % (landsat_file, str(datetime.now() - ssl_start)))

    return landsat_opath


# FUNCTION TO WRITE OUT CLASSIFIED RASTER
def write_geotiff(fname, data, geo_transform, projection, classes, COLORS, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(len(classes) + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0, AND BEN HICKSON',
        'TIFFTAG_DOCUMENTNAME': 'Land Cover Classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Random Forests Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):
    logger.error(txt, *args, **kwargs)
    exit(1)


def getQQuadFromNAIP(f):
    fname = os.path.basename(f)
    qquad = fname.split("_")[1] + "_" + fname.split("_")[2]
    return qquad


"""
def get_STDDev_VRT(naip_file, qquad_vrt_dir):
    naip_path = getFullNAIPPath(naip_file, naip_dir)
    qquad = getQQuadFromNAIP(naip_file)
    rasters_stddev = []

    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=3, overwrite=False)
    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=5, overwrite=False)
    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=10, overwrite=False)

    #for bandnum in range(1, 5):
    #    bandnum = "band" + str(bandnum)
    #    ffile = "stddev_" + os.path.splitext(naip_file)[0] + bandnum + ".tif"

    #   rasters_stddev.append(os.path.abspath(findFile(os.path.join(std3px_dir, bandnum), ffile)).replace("\\", "/"))
    #    rasters_stddev.append(os.path.abspath(findFile(os.path.join(std5px_dir, bandnum), ffile)).replace("\\", "/"))
    #    rasters_stddev.append(os.path.abspath(findFile(os.path.join(std10px_dir, bandnum), ffile)).replace("\\", "/"))

    stddev_vrt_dir = os.path.join(qquad_vrt_dir, "stddev")
    vrt_stddev = os.path.join(stddev_vrt_dir, qquad + "_stddev.vrt")
    #print(vrt_stddev)

    if not os.path.exists(vrt_stddev):
        build_vrt = "gdalbuildvrt -overwrite -separate %s %s" % (vrt_stddev, " ".join(rasters_stddev))
        logging.debug("BUILDING VRT WITH: \n\t%s" % build_vrt)
        os.system(build_vrt)

    return vrt_stddev

def get_GaussianFile(naip_file):
    #naip_path = getFullNAIPPath(naip_file, naip_dir)
    qquad = getQQuadFromNAIP(naip_file)

    gaussfile = gaussianCalc(naip_file, base_datadir, sigma=1, overwrite=False)

    return gaussfile
"""


def get_VegIndicies_VRT(naip_file, qquad_vrt_dir):
    print(naip_file)
    qquad = getQQuadFromNAIP(naip_file)
    rasters_float = []

    rasters_float.append(os.path.normpath(findVIFile("NDVI", ndvi_dir, naip_file)).replace("\\", "/"))
    rasters_float.append(os.path.normpath(findVIFile("SAVI", savi_dir, naip_file)).replace("\\", "/"))
    rasters_float.append(os.path.normpath(findVIFile("OSAVI", osavi_dir, naip_file)).replace("\\", "/"))
    rasters_float.append(os.path.normpath(findVIFile("MSAVI2", msavi2_dir, naip_file)).replace("\\", "/"))
    rasters_float.append(os.path.normpath(findVIFile("EVI2", evi2_dir, naip_file)).replace("\\", "/"))

    naipvis_vrt_dir = os.path.join(qquad_vrt_dir, "naipvis")
    vrt_naipvis = os.path.join(naipvis_vrt_dir, qquad + "_naipvis.vrt")

    if not os.path.exists(vrt_naipvis):
        build_vrt = "gdalbuildvrt -overwrite -separate %s %s" % (vrt_naipvis, " ".join(rasters_float))
        os.system(build_vrt)

    return vrt_naipvis


def createClassifiedFile(loc_NAIPFile, rf_classifier, rf_args, mltype="RF", overwrite=False):
    beg = datetime.now()
    print("Starting on qquad: %s" % loc_NAIPFile)
    logger.debug("Starting on qquad: %s" % loc_NAIPFile)

    file = os.path.basename(loc_NAIPFile)

    qquad = getQQuadFromNAIP(file)

    if mltype == "RF":
        output_fname = "{}_D{}E{}MPL{}_{}.tif". \
            format(mltype, rf_args["maxdepth"], rf_args["n_est"], rf_args["min_per_leaf"], qquad)
    else:
        print("Unknown classifier type '{}' specified. Exiting...")
        raise (ValueError)

    loc_classified_file = os.path.join(loc_classifiedQuarterQuads, output_fname)
    print("loc_classified_file; ", loc_classified_file)

    if not os.path.exists(loc_classified_file) or overwrite:
        cl_start = datetime.now()
        logging.info("\tClassifying landcover file at %s..." % (loc_classified_file))
        # loc_NAIPFile = os.path.join(root, file)

        """
        vrt_naipvis = get_VegIndicies_VRT(loc_NAIPFile, qquad_vrt_dir) #  All float32
        #vrt_stddev = get_STDDev_VRT(file)  # All 8 bit ("byte")
        gaussf_path = get_GaussianFile(loc_NAIPFile)

        # BEN!!!!!! YOU REMOVED LANDSAT VARIABLE FOR QUICK RUN
        landsat_path = createSubSetLandsat(loc_NAIPFile, landsat_file, landsat_qquad_dir).replace("\\", "/")

        landsat_ndsi_path = createSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_qquad_dir).replace("\\", "/")
        landsat_ndwi_path = createSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_qquad_dir).replace("\\", "/")

        # GET RASTER INFO FROM INPUT
        # NEED TO GET BANDS DATA INTO SINGLE ARRAY FOR OUTPUT CLASSIFICATION
        bands_data = []
        #for inras in [loc_NAIPFile, vrt_naipvis, gaussf_path, landsat_path, landsat_ndsi_path, landsat_ndwi_path]:
        for inras in [loc_NAIPFile, vrt_naipvis, gaussf_path, landsat_ndsi_path, landsat_ndwi_path]:
            print("\tIN RAS: {}".format(inras))
            try:
                raster_dataset = gdal.Open(inras, gdal.GA_ReadOnly)
            except RuntimeError as e:
                report_and_exit(str(e))

            geo_transform = raster_dataset.GetGeoTransform()
            proj = raster_dataset.GetProjectionRef()

            for b in range(1, raster_dataset.RasterCount + 1):
                band = raster_dataset.GetRasterBand(b)
                bands_data.append(band.ReadAsArray())
        """
        # GET PROJECTION INFO FOR GDAL WRITE
        try:
            raster_dataset = gdal.Open(loc_NAIPFile, gdal.GA_ReadOnly)
            geo_transform = raster_dataset.GetGeoTransform()
            proj = raster_dataset.GetProjectionRef()
        except RuntimeError as e:
            report_and_exit(str(e))

        training_raster = generateStack(loc_NAIPFile)
        # CREATE NP DATASTACK FROM ALL RASTERS
        # bands_data = np.dstack(bands_data)
        # CREATE VARIABLES OF ROWS, COLUMNS, AND NUMBER OF BANDS
        with rio.open(training_raster) as tras:
            print(tras)
            bands_data = tras.read()

        scikit_array = np.moveaxis(bands_data, 0, -1)
        print(scikit_array.shape)
        rows, cols, n_bands = scikit_array.shape
        n_samples = rows * cols

        # CREATE EMPTY ARRAY WITH SAME SIZE AS RASTER
        flat_pixels = scikit_array.reshape((n_samples, n_bands))

        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

        # A list of colors for each class
        COLORS = [
            "#000000",  # 0 EMPTY
            "#00af11",  # 1 - Vegetation - Thick
            "#00e513",  # 2 - Vegetation - Thin
            "#e9ff5a",  # 3 - Herbaceous
            "#00734C",  # 4 - Vegetation - Irrigated
            "#f1ac34",  # 5 - Barren - Light
            "#a9852e",  # 6 - Barren - Dark
            "#2759ff",  # 7 - Open Water - Shallow
            "#001866",  # 8 - Open Water - Deep
            "#efefef",  # 9 - Roof - White
            "#d65133",  # 10 - Roof - Red
            "#cecece",  # 11 - Roof - Grey
            "#a0a0a0",  # 12 - Impervious - Light
            "#555555",  # 13 - Impervious - Dark
            "#000000",  # 14 - Shadows
        ]

        print("Classifying...")
        """if not np.all(np.isfinite(flat_pixels)):
            print("Not all value finite. Fixing...")
            flat_pixels = np.where(flat_pixels > np.finfo(np.float32).max, np.finfo(np.float32).max, flat_pixels)
        if np.any(np.isnan(flat_pixels)):
            print("Some values are NaN. Fixing...")
            flat_pixels = np.where(flat_pixels == np.NaN, np.finfo(np.float32).max, flat_pixels)
        """
        # try:
        result = rf_classifier.predict(flat_pixels)
        # Reshape the result: split the labeled pixels into rows to create an image
        classification = result.reshape((rows, cols))

        # WRITE OUT THE CLASSIFIED ARRAY TO RASTER BASED ON PROPERTIES OF TRAINING RASTERS
        # TODO - Rewrite this to use rasterio for consistency
        write_geotiff(loc_classified_file, classification, geo_transform, proj, classes, COLORS)
        logging.info("\tCreated classified file in %s" % (str(datetime.now() - cl_start)))
        # except (ValueError) as e:
        # logging.info("-----------BAD VALUES FOR PREDICTORS. SKIPPING FILE %s\n%s" % (file, str(e)))
        # return None

        del bands_data
        del flat_pixels

    else:
        logging.info("LandCover file %s exists and no overwrite." % loc_classified_file)

    """
    o_veg_loc = r"Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\Data\RiparianClass_VBs"
    utilities.useDirectory(o_veg_loc)

    quadrant_loc = os.path.join(o_veg_loc, qquad[:5])
    utilities.useDirectory(quadrant_loc)

    # not writing out density calculations
    #dq_path = os.path.join(degress_quadrant_loc, denseVeg_file)
    #sq_path = os.path.join(degress_quadrant_loc, sparseVeg_file)

    riparian_class_qquad = os.path.join(quadrant_loc, "RiparianClassification_" + qquad + ".tif")

    if not os.path.exists(riparian_class_qquad) or overwrite:
        createRiparianClass(loc_classified_file, riparian_class_qquad, qquad)

    shutil.copy(riparian_class_qquad, r"Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\Data\Pinal_AOI")
    """

    logger.debug("COMPLETED riparian classification. Finished in %s" % (str(datetime.now() - beg)))
    logger.debug("_____________________________________________________________________________________")

    return loc_classified_file


def createRiparianClass(lc_raster, o_file, qquad):
    rc_start = datetime.now()
    logging.info("\tClassifying riparian zones for %s" % lc_raster)
    with rio.open(lc_raster) as class_file:
        class_array = class_file.read(1)  # _band(1)
        kwargs = class_file.profile

    # Get average densities of each class across the whole raster.
    # TO DO - Updatet this to evaluate on something more specific than the qquad area
    dense_veg_array = np.where(class_array == 1, 1, 0)
    dense_file_avg = np.mean(dense_veg_array)
    sparse_veg_array = np.where(class_array == 2, 1, 0)
    sparse_file_avg = np.mean(sparse_veg_array)

    sparse_veg_array_localmean = ndimage.uniform_filter(sparse_veg_array.astype(np.float32), size=vaa_diameter,
                                                        mode='constant')
    dense_veg_array_localmean = ndimage.uniform_filter(dense_veg_array.astype(np.float32), size=vaa_diameter,
                                                       mode='constant')

    # ------------------------------------------------
    # CRITICAL : Identify the splits where xero, meso, and hydro will be identified
    # based on density ofsparse and thick vegetation
    sparse_xero_lowlimit = sparse_file_avg + np.std(sparse_veg_array_localmean)
    sparse_meso_lowlimit = sparse_file_avg + (1 - sparse_file_avg) * 0.7
    sparse_hydro_lowlimit = sparse_file_avg + (1 - sparse_file_avg) * 0.9

    dense_xero_lowlimit = dense_file_avg + np.std(dense_veg_array_localmean)
    dense_meso_lowlimit = dense_file_avg + (1 - dense_file_avg) * 0.7
    dense_hydro_lowlimit = dense_file_avg + (1 - dense_file_avg) * 0.9

    # ------------------------------------------------

    # Reassign pixel values based on density assessment
    sparse_local_xero = np.where(sparse_veg_array_localmean > sparse_xero_lowlimit, 1,
                                 0)  # xero (1) if true, upland (0) if false
    sparse_local_meso = np.where(sparse_veg_array_localmean > sparse_meso_lowlimit, 2,
                                 0)  # meso (2) if true, upland (0) if false
    sparse_local_hydro = np.where(sparse_veg_array_localmean > sparse_hydro_lowlimit, 3,
                                  0)  # hydro (3) if true, upland (0) if false
    # For some reason can't take numpy.maximum from more than two arrays at once
    sparse_combine = np.maximum(sparse_local_xero, sparse_local_meso)  # , sparse_local_hydro)
    sparse_combine = np.maximum(sparse_combine, sparse_local_hydro)

    dense_local_xero = np.where(dense_veg_array_localmean > dense_xero_lowlimit, 1,
                                0)  # xero (1) if true, upland (0) if false
    dense_local_meso = np.where(dense_veg_array_localmean > dense_meso_lowlimit, 2,
                                0)  # meso (2) if true, upland (0) if false
    dense_local_hydro = np.where(dense_veg_array_localmean > dense_hydro_lowlimit, 3,
                                 0)  # hydro (3) if true, upland (0) if false
    # For some reason can't take numpy.maximum from more than two arrays at once
    dense_combine = np.maximum(dense_local_xero, dense_local_meso)  # , sparse_local_hydro)
    dense_combine = np.maximum(dense_combine, dense_local_hydro)

    # COMPARISON OF DENSITY VALUES OF BOTH RASTERS AT EACH PIXEL FOR
    # DETERMINATION. ESSENTAILLY A DECISION TREE
    p = np.where(dense_combine == 0, np.where(sparse_combine == 0, 0, 0), 0)
    o = np.where(dense_combine == 0, np.where(sparse_combine == 1, 1, p), p)
    n = np.where(dense_combine == 0, np.where(sparse_combine == 2, 2, o), o)
    m = np.where(dense_combine == 0, np.where(sparse_combine == 3, 3, n), n)
    l = np.where(dense_combine == 1, np.where(sparse_combine == 0, 1, m), m)
    k = np.where(dense_combine == 1, np.where(sparse_combine == 1, 1, l), l)
    j = np.where(dense_combine == 1, np.where(sparse_combine == 2, 2, k), k)
    i = np.where(dense_combine == 1, np.where(sparse_combine == 3, 3, j), j)
    h = np.where(dense_combine == 2, np.where(sparse_combine == 0, 1, i), i)
    g = np.where(dense_combine == 2, np.where(sparse_combine == 1, 2, h), h)
    f = np.where(dense_combine == 2, np.where(sparse_combine == 2, 2, g), g)
    e = np.where(dense_combine == 2, np.where(sparse_combine == 3, 3, f), f)
    d = np.where(dense_combine == 3, np.where(sparse_combine == 0, 2, e), e)
    c = np.where(dense_combine == 3, np.where(sparse_combine == 1, 2, d), d)
    b = np.where(dense_combine == 3, np.where(sparse_combine == 2, 3, c), c)
    riparian = np.where(dense_combine == 3, np.where(sparse_combine == 3, 3, b), b)

    kwargs.update(
        dtype=np.uint8,
        nodata=0,
        compress='lzw'
    )

    valleybottom_ras = findVBRaster(qquad)

    with rio.open(valleybottom_ras) as vb_raster:
        vb_array = vb_raster.read(1).astype(np.float32)

    # print("Clipping to Valley Bottoms")
    clipped_riparian = np.where(vb_array > 1, riparian, 0)

    with rio.open(o_file, 'w', **kwargs) as dst:
        dst.write_band(1, clipped_riparian.astype(np.uint8))

        dst.write_colormap(
            1, {
                0: (255, 255, 255),
                1: (186, 228, 179),
                2: (116, 196, 118),
                3: (35, 139, 69)})
        cmap = dst.colormap(1)

    logging.info("\tFinished riparian classification in %s" % (str(datetime.now() - rc_start)))


def findVBRaster(qquad, overwrite=False):
    vb_start = datetime.now()
    logging.debug("Starting creation of subset of valley bottom...")
    naip_path = getFullNAIPPath(qquad, naip_dir)
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


def apply_and_concat(dataframe, field, func, column_names):
    return pd.concat((
        dataframe,
        dataframe[field].apply(
            lambda cell: pd.Series(func(cell), index=column_names))), axis=1)


def calculateGeom(row):
    # print(row)
    geom = row["geometry"]
    if row['PROJ'] == "NAD83 / UTM zone 11N":
        x = geom.centroid.x
        y = geom.centroid.y

        point = Point(transform(utm12, utm11, x, y))
        if point.x <= 0 or point.y <= 0:
            print("Bad", point, "Original: ", geom)
        return point
    else:
        return geom


def extractToPoints(training_points, out_file, data_dir, landsatdir, overwrite=False):
    def get_values(geom):
        # print(row)
        # geom = row['geometry']
        x = geom.centroid.x
        y = geom.centroid.y

        values = []
        # for raster in raster_objects:
        # print("Starting Raster Extract for %s at x:%s y:%s" % (os.path.basename(raster), str(x), str(y)))
        # with rio.open(raster) as ras:
        """for val in rasnaip.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        for val in rasnaipvis.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        for val in rasgauss.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        for val in raslandsat.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        for val in rasNDSI.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        for val in rasNDWI.sample([(x, y)]):
            values += np.ndarray.tolist(val)
        """
        # print(geom)
        for val in ras_stack.sample([(x, y)]):
            values += np.ndarray.tolist(val)

        return pd.Series(values, index=band_order)

    ext_start = datetime.now()
    landsat_qquad_dir = os.path.join(landsatdir, "byNAIPDOY_QQuads")
    ndwi_qquad_dir = os.path.join(data_dir, "NDWI")
    ndsi_qquad_dir = os.path.join(data_dir, "NDSI")
    # LOCATION OF THE NDSI FILE
    ndsi_file = os.path.join(ndsi_qquad_dir, "LandsatOLI_NDSI_30m.tif")
    # LOCATION OF THE NDWI FILE
    ndwi_file = os.path.join(ndwi_qquad_dir, "LandsatOLI_NDWI_30m.tif")

    utils.useDirectory(landsat_qquad_dir)
    utils.useDirectory(ndwi_qquad_dir)
    utils.useDirectory(ndsi_qquad_dir)

    # IF VECTOR FILE OF POINTS WITH RASTER EXTRACTS DOESN'T EXIST, BUILD IT
    if not os.path.exists(out_file) or overwrite:
        # if "class_points" not in locals():
        #    logging.debug("READING IN %s as training_points" % training_points)
        training_data_df = gpd.read_file(training_points, crs={'init': 'epsg:26912'})
        # print("Columns: ", training_data_df.columns)

        if "utm_geom" not in training_data_df:
            logging.debug("ADDING COLUMN 'utm_geom' WITH CORRECT UTM COORDINATES FOR EACH QUARTER QUAD")
            # CREATE TRUE RASTER GEOMETRY COLUMN (BASED ON UTM)
            training_data_df["utm_geom"] = training_data_df.apply(calculateGeom, axis=1)

        # NDSI is only used because its the last raster column
        if "L8_NDWI" not in training_data_df:
            logging.debug("CREATING COLUMNS...")
            # CREATE EMPTY COLUMNS IN DATA FRAME FOR EACH RASTER VARIABLE

            sample_naip = training_data_df.iloc[0]['NAIP_FILE']
            raster_loc = generateStack(sample_naip)
            band_columns = []
            with rio.open(raster_loc) as ras_stack:
                for i in range(1, ras_stack.count + 1):
                    column = ras_stack.tags(i)['NAME']
                    band_columns.append(column)
                    training_data_df[column] = np.NaN
        # for column in band_columns:


        # print("COLUMNS", training_data_df.columns)

        net_percentage = 0.0
        # ITERATE THROUGH DATAFRAME IN GROUPS BY NAIP_FILE. KEEPS US FROM OPENING/CLOSING RASTERS FOR EACH POINT - INSTEAD FOR EACH GROUP
        for loc_NAIPFile, group in training_data_df.groupby("NAIP_FILE"):
            logger.debug("\nStarting raster value extraction for points in qquad %s" % loc_NAIPFile)
            print("\nStarting raster value extraction for points in qquad %s" % loc_NAIPFile)
            loc_NAIPFile.replace("\\", "/")  # normalize for windows paths

            # LOOK FOR RASTERS FROM WHICH VALUES WILL BE EXTRACTED
            # file = os.path.basename(loc_NAIPFile)

            """
            vrt_naipvis = get_VegIndicies_VRT(loc_NAIPFile, qquad_vrt_dir)
            # vrt_stddev = get_STDDev_VRT(file)
            gaussf_path = get_GaussianFile(loc_NAIPFile)

            landsat_path = createSubSetLandsat(loc_NAIPFile, landsat_file, landsat_qquad_dir).replace("\\", "/")

            landsat_ndsi_path = createSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_qquad_dir).replace("\\", "/")
            landsat_ndwi_path = createSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_qquad_dir).replace("\\", "/")

            net_percentage += 100 * len(training_data_df.loc[training_data_df["NAIP_FILE"] == loc_NAIPFile]) / len(
                training_data_df)
            logger.debug("Percentage of total: %d" % net_percentage)
            # SELECT POINTS WHICH HAVE NAIP PATH VALUE

            # Only if group hasn't had values assigned (Jupyter and Rodeo iterations)
            """

            if group["L8_NDWI"].isnull().values.any():
                training_raster = generateStack(loc_NAIPFile)

                band_order = []
                with rio.open(training_raster) as ras_stack:
                    for i in range(1, ras_stack.count + 1):
                        band_order.append(ras_stack.tags(i)['NAME'])

                    training_data_df.loc[
                        training_data_df.NAIP_FILE == loc_NAIPFile, band_columns] = \
                        training_data_df.loc[training_data_df.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(get_values)

            """
            if group["NDSI"].isnull().values.any():
                with rio.open(loc_NAIPFile) as rasnaip:
                    with rio.open(vrt_naipvis) as rasnaipvis:
                        with rio.open(gaussf_path) as rasgauss:
                            with rio.open(landsat_path) as raslandsat:
                                with rio.open(landsat_ndsi_path) as rasNDSI:
                                    with rio.open(landsat_ndwi_path) as rasNDWI:
                                        count = 0
                                        training_data_df.loc[
                                            training_data_df.NAIP_FILE == loc_NAIPFile, rasters_names] = \
                                            training_data_df.loc[
                                                training_data_df.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(
                                                get_values)
            """
            logger.debug("Finished with group %s at %s" % (loc_NAIPFile, str(datetime.now())))

        # REMOVE ALL ROWS WHICH EXTRACTED NO DATA VALUES FROM LANDSAT
        # for column in landsat:
        #    class_points = class_points[class_points.loc[column] != 32766]

        logger.info("Finished raster value extraction of %s points in %s" % (
            str(len(training_data_df)), str(datetime.now() - ext_start)))

        # GEOPANDAS WON"T ALLOW MORE THAN ONE COLUMN WITH GEOMETRY TYPE. REMOVE THE utm_geom COLUMN CREATED PREVIOUSLY
        del training_data_df['utm_geom']
        # print("COLUMNS:\n", training_data_df.columns)
        print("WRITING DATAFRAME TO OUTPUT...")
        logger.debug("WRITING DATAFRAME TO OUTPUT...")
        training_data_df.to_file(out_file)

    else:
        # if "training_data_df" not in "locals":
        print("Reading in point file %s" % out_file)
        logger.info("Reading in point file %s" % out_file)

        # BEN - UPDATE RASTERS NAMES READ
        training_data_df = gpd.read_file(out_file)
        # Had to delete utm_geom when writing file (can't have two geometry columns). Recreate...
        print(training_data_df.columns)
        band_columns = training_data_df.columns.tolist()[4:-1]

    return {"training_points": training_data_df, "band_names": band_columns}


def getClassifier(classifier_file, train_data, rf_rasters, args, createClassifier=False):
    """ Either returns classifier from file or creates a new one """
    if not os.path.exists(classifier_file) or createClassifier:
        # save classifier to file
        # TRAIN RANDOM FORESTS
        rf_start = datetime.now()
        logger.info("Beginning Random Forest Train")
        # maxdepth = 400
        # n_est = 40
        # n_job = 1
        # min_per_leaf = 50
        # crit = "entropy"  # gini or entropy

        rf_model = RandomForestClassifier(verbose=1, max_depth=args["maxdepth"], n_estimators=args["n_est"],
                                          n_jobs=args["n_job"], min_samples_leaf=args["min_per_leaf"],
                                          criterion=args["crit"])

        #X = Imputer().fit_transform(train_data[rf_rasters].dropna())

        rf_model.fit(train_data[rf_rasters].dropna(),
                     train_data[rf_rasters + ["Class"]].dropna()["Class"])

        logger.info("Finished Fitting in", datetime.now() - rf_start)
        _ = joblib.dump(rf_model, classifier_file, compress=9)
    else:
        # load classifier from file
        rf_model = joblib.load(classifier_file)

    return rf_model


def createClassification(base_datadir, naip_dir=False):
    if not naip_dir:
        naip_dir = os.path.join(base_datadir, "NAIP")

    base_landsatdir = os.path.join(base_datadir, "Landsat8")

    training_data_dir = os.path.join(base_datadir, "inital_model_inputs")

    # ndvi_dir = utils.useDirectory(os.path.join(base_datadir, "NDVI"))
    # savi_dir = utils.useDirectory(os.path.join(base_datadir, "SAVI"))
    # osavi_dir = utils.useDirectory(os.path.join(base_datadir, "OSAVI"))
    # msavi2_dir = utils.useDirectory(os.path.join(base_datadir, "MSAVI2"))
    # evi2_dir = utils.useDirectory(os.path.join(base_datadir, "EVI2"))
    ndwi_qquad_dir = utils.useDirectory(os.path.join(base_datadir, "NDWI"))
    ndsi_qquad_dir = utils.useDirectory(os.path.join(base_datadir, "NDSI"))

    # std3px_dir = os.path.join(base_datadir, "StdDev_3px")
    # std5px_dir = os.path.join(base_datadir, "StdDev_5px")
    # std10px_dir = os.path.join(base_datadir, "StdDev_10px")

    base_landsatdir = utils.useDirectory(os.path.join(base_datadir, "Landsat8"))
    # landsat_qquad_dir = utils.useDirectory(os.path.join(base_landsatdir, "byNAIPDOY_QQuads"))
    valleybottoms_dir = utils.useDirectory(os.path.join(base_datadir, "ValleyBottoms"))
    loc_valleybottoms = utils.useDirectory(os.path.join(valleybottoms_dir, "VBET_ValleyBottoms"))

    # LOCATION OF THE LANDSAT FILE
    landsat_file = os.path.os.path.join(base_landsatdir, "Landsat1to8_TOA_NAIPAcquiDate_merge_rectified.tif")
    # LOCATION OF THE NDSI FILE
    ndsi_file = os.path.join(ndsi_qquad_dir, "LandsatOLI_NDSI_30m.tif")
    # LOCATION OF THE NDWI FILE
    ndwi_file = os.path.join(ndwi_qquad_dir, "LandsatOLI_NDWI_30m.tif")

    # LOCATION OF FILE CONTAINING CLASSIFICATION POLYGONS
    loc_class_polygons = os.path.join(training_data_dir, 'classificationTrainingPolygons.shp')

    # LOCATION OF USGS QUARTER QUADs in ARIZONA
    loc_usgs_qquads = os.path.join(training_data_dir, "USGS_QQuads_AZ.shp")

    # DIRECTORY HOLDING VRTS BY QUARTER QUAD FOR VEGETATION INDICIES (FLOAT32) AND STD_DEV (UINT8))
    qquad_vrt_dir = os.path.join(base_datadir, "QQuad_VRTs")

    day = datetime.today().strftime('%Y%m%d')
    global loc_classifiedQuarterQuads
    loc_classifiedQuarterQuads = utils.useDirectory(os.path.join(base_datadir, "classifiedQuarterQuads_" + day))
    print("loc_classifiedQuarterQuads: ", loc_classifiedQuarterQuads)

    loc_classifier = os.path.join(training_data_dir, "RandomForestClassifier.joblib.pkl")

    aoi = gpd.read_file(os.path.join(training_data_dir, "Ecoregions_AOI.gpkg"))
    aoi = gpd.read_file(os.path.join(training_data_dir, "PinalCounty_AOI.gpkg"))

    valleybottoms_dir = utils.useDirectory(os.path.join(base_datadir, "ValleyBottoms"))
    loc_valleybottoms = utils.useDirectory(os.path.join(valleybottoms_dir, "VBET_ValleyBottoms"))
    VBET_VB_loc = os.path.join(valleybottoms_dir, "VBET_ValleyBottoms_20180624.tif")

    # DEFINE THE PROJECTION USED OVER ARIZONA. USED FOR TRANSLATING POINT GEOMETRY LATER ON
    utm11 = Proj(init="epsg:26911")
    utm12 = Proj(init="epsg:26912")

    """
    # IDENTIFY RASTER VARIABLES
    # THESE ORDER OF THESE RASTER VARIABLES MUST COINCIDE WITH THE CONSTRUCTED ARRAY OF EXTRACTS in the get_values FUNCTION
    naip = ["NAIP1", "NAIP2", "NAIP3", "NAIP4"]
    landsat = ["Landsat1", "Landsat2", "Landsat3", "Landsat4", "Landsat5", "Landsat6", "Landsat7", "Landsat8"]
    landsat_vis = ["NDSI", "NDWI"]
    naip_vis = ["NDVI", "EVI2", "SAVI", "OSAVI", 'MSAVI2']

    textures = ["StdDev_3px_band1", "StdDev_3px_band2", "StdDev_3px_band3", "StdDev_3px_band4",
                "StdDev_5px_band1", "StdDev_5px_band2", "StdDev_5px_band3", "StdDev_5px_band4",
                "StdDev_10px_band1", "StdDev_10px_band2", "StdDev_10px_band3", "StdDev_10px_band4"]
    filters = ["Gauss1_band1", "Gauss1_band2", "Gauss1_band3", "Gauss1_band4",]
    #rasters_names = naip + naip_vis + filters + landsat + landsat_vis
    #print(rasters_names)
    """

    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    loc_out_join = Create_Classification_Points.joinProjectionAndNAIP(loc_class_polygons, naip_dir, loc_usgs_qquads,
                                                                      overwrite=False)

    loc_points_in_poly = Create_Classification_Points.createShapefilePoints(loc_out_join, overwrite=False)

    # LOCATION OF FILE CONTAINING CLASSIFICATION POINTS POST EXTRACT
    loc_points_wRaster_extracts = loc_points_in_poly[:-4] + "_extracts.shp"

    # EXTRACT RASTER VALUES TO POINTS
    training_info = extractToPoints(loc_points_in_poly, loc_points_wRaster_extracts, base_datadir, base_landsatdir,
                                    overwrite=False)

    class_points = training_info["training_points"]
    band_names = training_info["band_names"]
    logger.debug("Available raster variables: \n\t%s" % band_names)

    class_points["utm_geom"] = class_points.apply(calculateGeom, axis=1)

    # Split the points data frame into train and test
    training_data, testing_data = train_test_split(class_points, test_size=0.3)

    # INITIALIZE COLUMN FOR PREDICTED CLASSIFICATION VALUES
    predicted_column = "CLASS_PREDICT"
    testing_data[predicted_column] = "Null"

    # remove_landsat = True

    # rasters values used in random forest
    # Allows removal of some rasters
    temp_rasters = band_names[:]
    rf_rasters = band_names[:]
    print("rf_raster: ", rf_rasters)

    """
    if remove_landsat:
        for r in temp_rasters:
            if "Landsat" in r:
                rf_rasters.remove(r)
    """

    logger.info("Using raster variables: \n%s" % rf_rasters)

    rf_args = {"maxdepth": 250,
               "n_est": 100,
               "n_job": 2,
               "min_per_leaf": 50,
               "crit": "entropy"}  # gini or entropy}

    # print('Out-of-bag score estimate:', {rf.oob_score_:.3})
    rf = getClassifier(loc_classifier, training_data, rf_rasters, rf_args, createClassifier=False)

    irods_sess, irods_files = utils.getFilesonDE()

    print("-------------------STARTING CREATION FOR AOI-----------------------------")
    # aoi.crs = fiona.crs.from_epsg(2163)
    print(aoi.crs)

    footprints = gpd.read_file(loc_usgs_qquads)
    aoi.to_crs(footprints.crs, inplace=True)

    aoi_qquads = []
    for j, arow in aoi.iterrows():
        for i, row in footprints.iterrows():
            if row.geometry.intersects(arow.geometry):
                fpath = getFullNAIPPath(row.QUADID, naip_dir)
                aoi_qquads.append(fpath)

    print(len(aoi_qquads))
    print("AOI QQUADS\n{}\n".format(aoi_qquads))
    if len(aoi_qquads) == 0:
        print("Error - no qquads found intersecting AOI. Exiting")
        raise Exception

    Parallel(n_jobs=3)(
        delayed(createClassifiedFile)(naip_file, rf, rf_args, overwrite=True) for naip_file in aoi_qquads)
    # beg = datetime.now()
    # logger.debug("Starting on qquad: %s" % qquad_name)
    # createClassifiedFile(fpath, rf, rf_args, overwrite=False)
    # logger.debug("COMPLETED riparian classification. Finished in %s" % (str(datetime.now() - beg)))
    # logger.debug("_____________________________________________________________________________________")

    print("-------------------FINISHED CREATING FOR AOI-----------------------------")

    print("------------------- STARTING CREATION FOR TRAINING QUADS -----------------------------")
    files_list = []
    for loc_NAIPFile, group in class_points.groupby("NAIP_FILE"):
        files_list.append(loc_NAIPFile)

    Parallel(n_jobs=2)(delayed(createClassifiedFile)(naip_file, rf, rf_args) for naip_file in files_list)
    #    classified_file_rf = createClassifiedFile(loc_NAIPFile, rf, rf_args, overwrite=False)

    print("-------------------STARTING CREATION FOR SUB AREA-----------------------------")
    single_comp_subset = []
    with open(os.path.join(base_datadir, "NaipDone1.txt")) as txt:
        for l in txt:
            fullname = getFullNAIPPath(l[:-1], naip_dir)
            # print(l)
            single_comp_subset.append(fullname)
            # if l[:-1] not in irods_files.keys():
            #    single_comp_subset.append(fullname)

            # createClassifiedFile(fullname, rf, rf_args)

    Parallel(n_jobs=2)(delayed(createClassifiedFile)(naip_file, rf, rf_args) for naip_file in single_comp_subset)
    # for qquad_name in single_comp_subset:
    # fpath = getFullNAIPPath(qquad_name, naip_dir)

    # createClassifiedFile(fpath, "RF", rf, rf_args, overwrite=False)

    print("-------------------FINISHED CREATION FOR SUB AREA-----------------------------")

    exit()

    def get_class_value(geom):
        """ TAKES A VARIABLE OF GEOMETRY TYPE AND RETURNS THE VALUE AT X,Y
        AS A PANDAS SERIES FOR FOR LOCAL RASTER 'CLASSRAS' """
        # print(row)
        # geom = row.geometry
        x = geom.centroid.x
        y = geom.centroid.y
        for val in classras.sample([(x, y)]):
            # print(np.ndarray.tolist(val))
            return pd.Series(val, index=[predicted_column])

    # CREATE CLASSIFIED RASTERS FOR QUARTER QUADS USED IN TRAINING DATA FIRST
    for loc_NAIPFile, group in class_points.groupby("NAIP_FILE"):
        classified_file_rf = createClassifiedFile(loc_NAIPFile, rf, rf_args, overwrite=False)

        # EXTRACT PREDICTED PIXEL CLASSIFICATION TO TESTING DATAFRAME
        print("Extracting predicted classified values...")
        with rio.open(classified_file_rf) as classras:
            # print(classras.indexes)
            testing_data.loc[testing_data.NAIP_FILE == loc_NAIPFile, [predicted_column]] = \
                testing_data.loc[testing_data.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(get_class_value)

    """THIS CODE BLOCK USES A SHAPEFILE OF THE NAIP FOOTPRINTS AND THE AOI ECOREGIONS FEATURE
    CLASS TO FIND ONLY THE NAMES OF THE QQUADS WHICH WE WAND TO CLASSIFY. THEN IDENTIFIES THE
    ACTUAL PATH OF THE NAIP FILE AND PASSES IT TO THE CLASSIFIER"""

    # THEN CREATE CLASSIFIED RASTER FROM ALL OTHER QUARTER QUADS
    for root, dirs, files in os.walk(naip_dir):
        for file in files:
            if file.endswith(".tif"):
                fpath = os.path.join(root, file)
                # createClassifiedFile(fpath, rf, rf_args, overwrite=False)

    logger.info("\n\t\t-- DONE WITH ALL FILES --\n")

    # ----------------------------------------------------
    # # TESTING SECTION


    """flat_pixels = np.array([1,10,50,500, np.NaN])

    if np.isfinite(flat_pixels.any()) and not np.isnan(flat_pixels.any()):
        flat_pixels = np.where(flat_pixels > np.finfo(np.float32).max, np.finfo(np.float32).max, flat_pixels)
        #result = rf_classifier.predict(flat_pixels)
        #np.where(x.values >= np.finfo(np.float32).max,)
        print("MODING")
    else:
        print("TURRIBLE")"""


utm11 = Proj(init="epsg:26911")
utm12 = Proj(init="epsg:26912")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    riolog = rio.logging.getLogger()
    riolog.setLevel(logging.ERROR)

    start = datetime.now()

    overwrite = True

    global qquad, naip_dir, loc_valleybottoms, VBET_VB_loc, \
        ndvi_dir, savi_dir, osavi_dir, msavi2_dir, evi2_dir, loc_classifiedQuarterQuads

    veg_assessment_area = 0.1  # in acres
    vaa_meters = veg_assessment_area * 4046.86
    vaa_radius = math.sqrt(vaa_meters / math.pi)
    vaa_diameter = vaa_radius * 2

    # LOCATIONS OF FOLDERS HOLDING ALL INPUT RASTER DATA
    naip_data_dir = os.path.abspath(r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed")

    data_dir = os.path.abspath(r"M:\Data")

    createClassification(data_dir, naip_data_dir)

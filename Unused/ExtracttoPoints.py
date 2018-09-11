import logging
import geopandas as gpd
import rasterio as rio
import numpy as np
import fiona
from datetime import datetime
import os
import pandas as pd
import gdal
from pyproj import transform, Proj

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from shapely.geometry import Point

os.chdir(r"Q:/Arid Riparian Project/Scripts")

import Utilities as utilities
from RasterCalculations import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start = datetime.now()

overwrite = True

"""
def oldcreateSubSetLandsat(naip_path, landsat_vrt):
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

    ofile = "Landsat8_" + os.path.basename(naip_path)

    landsat_opath = os.path.join(dir, ofile)
    gdal_warp = "gdalwarp -overwrite -tap -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
        proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_vrt, landsat_opath)
    print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_vrt, naip_path))
    os.system(gdal_warp)

    print("Finished with warp in", datetime.now() - start)
"""

def getFullNAIPPath(naip_file, naipdir):
    for root,dirs,files in os.walk(naipdir):
        for file in files:
            if file == naip_file:
                return os.path.join(root,file)
    
    print("Unable to find naip file %s. Exiting" % naip_file)
    exit()


def findSTDDevFile(dir, naip_file, band_num, windowsize):
    #findFile(os.path.join(std3px_dir, bandnum), ffile)
    
    window_dir = os.path.join(dir, "StdDev_" + str(windowsize) + "px")
    utilities.useDirectory(window_dir)
    band_dir = os.path.join(window_dir, "band" + band_num)
    utilities.useDirectory(band_dir)
    
    for root, dirs, files in os.walk(band_dir):
        for file in files:
            if f in file:
                fpath = os.path.join(root, file)
                return fpath
                
    if "fpath" not in locals():
        standardDeviation(naip_file, dir, window_size=windowsize, overwrite=False)
    print("Unable to find file %s" % f)
    exit()


def findVIFile(type, dir, f):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if f in file:
                fpath = os.path.join(root, file)
                return fpath

    if "fpath" not in locals():
        vegIndexCalc(fpath, dir, [type])
    return None


def createSubSetLandsat(naip_path, landsat_file, opath):
    ofile = "Landsat8_" + os.path.basename(naip_path)

    landsat_opath = os.path.join(opath, ofile)
    
    if os.path.exists(landsat_opath) and not overwrite:
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

        if "ndsi" in opath.lower() or "ndwi" in opath.lower():
            resampletype = "bilinear"
        else:
            resampletype = "bilinear"
            #resampletype = "near"

        gdal_warp = "gdalwarp -overwrite -tap -r %s -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
            resampletype, proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), landsat_file, landsat_opath)
        print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_file, naip_path))
        os.system(gdal_warp)

        print("Finished with warp in", datetime.now() - start)

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
    for pixel_value in range(len(classes)+1):
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


def get_values(geom):
    #print(row)
    #geom = row['geometry']
    x = geom.centroid.x
    y = geom.centroid.y

    values = []
    # for raster in raster_objects:
    # print("Starting Raster Extract for %s at x:%s y:%s" % (os.path.basename(raster), str(x), str(y)))
    # with rio.open(raster) as ras:
    for val in rasnaip.sample([(x, y)]):
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
    
    return pd.Series(values, index=rasters_names)


def getQQuadFromNAIP(f):
    qquad = f.split("_")[1] + "_" + f.split("_")[2]
    return qquad


def get_STDDev_VRT(naip_file):
    naip_path = getFullNAIPPath(naip_file, naip_dir)
    qquad = getQQuadFromNAIP(naip_file)
    rasters_stddev = []
    
    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=3, overwrite=False)
    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=5, overwrite=False)
    rasters_stddev += standardDeviation(naip_path, base_datadir, window_size=10, overwrite=False)
    """
    for bandnum in range(1, 5):
        bandnum = "band" + str(bandnum)
        ffile = "stddev_" + os.path.splitext(naip_file)[0] + bandnum + ".tif"
        
        rasters_stddev.append(os.path.abspath(findFile(os.path.join(std3px_dir, bandnum), ffile)).replace("\\", "/"))
        rasters_stddev.append(os.path.abspath(findFile(os.path.join(std5px_dir, bandnum), ffile)).replace("\\", "/"))
        rasters_stddev.append(os.path.abspath(findFile(os.path.join(std10px_dir, bandnum), ffile)).replace("\\", "/"))
    """
    stddev_vrt_dir = os.path.join(qquad_vrt_dir, "stddev")
    vrt_stddev = os.path.join(stddev_vrt_dir, qquad + "_stddev.vrt")
    #print(vrt_stddev)

    if not os.path.exists(vrt_stddev):
        build_vrt = "gdalbuildvrt -overwrite -separate %s %s" % (vrt_stddev, " ".join(rasters_stddev))
        #print("BUILDING VRT WITH: ", build_vrt)
        os.system(build_vrt)

    return vrt_stddev


def get_GaussianFile(naip_file):
    naip_path = getFullNAIPPath(naip_file, naip_dir)
    qquad = getQQuadFromNAIP(naip_file)
    
    gaussfile = gaussianCalc(naip_path, base_datadir, sigma=1, overwrite=False)
    
    return gaussfile
    
    
def get_VegIndicies_VRT(naip_file):
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


def createClassifiedFile(loc_NAIPFile, rf_classifier):
    file = os.path.basename(loc_NAIPFile)

    qquad = getQQuadFromNAIP(file)

    output_fname = "Classified_" + "D" + str(maxdepth) + "E" + str(n_est) + "MPL" + str(min_per_leaf) + "_" + qquad + ".tif"
    loc_classified_file = os.path.join(loc_classifiedQuarterQuads, output_fname)

    if not os.path.exists(loc_classified_file) or overwrite:
        cl_start = datetime.now()
        print("CREATING OUTPUT FILE %s..." % (loc_classified_file))
        # loc_NAIPFile = os.path.join(root, file)

        file = os.path.basename(loc_NAIPFile)

        vrt_naipvis = get_VegIndicies_VRT(file) #  All float32
        #vrt_stddev = get_STDDev_VRT(file)  # All 8 bit ("byte")
        gaussf_path = get_GaussianFile(file)

        landsat_path = createSubSetLandsat(loc_NAIPFile, landsat_file, landsat_dir).replace("\\", "/")

        landsat_ndsi_path = createSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_dir).replace("\\", "/")
        landsat_ndwi_path = createSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_dir).replace("\\", "/")

        # GET RASTER INFO FROM INPUT
        # NEED TO GET BANDS DATA INTO SINGLE ARRAY FOR OUTPUT CLASSIFICATION
        bands_data = []
        for inras in [loc_NAIPFile, vrt_naipvis, gaussf_path, landsat_path, landsat_ndsi_path, landsat_ndwi_path]:
            try:
                raster_dataset = gdal.Open(inras, gdal.GA_ReadOnly)
            except RuntimeError as e:
                report_and_exit(str(e))

            geo_transform = raster_dataset.GetGeoTransform()
            proj = raster_dataset.GetProjectionRef()

            for b in range(1, raster_dataset.RasterCount + 1):
                band = raster_dataset.GetRasterBand(b)
                bands_data.append(band.ReadAsArray())

        # CREATE NP DATASTACK FROM ALL RASTERS
        bands_data = np.dstack(bands_data)
        # CREATE VARIABLES OF ROWS, COLUMNS, AND NUMBER OF BANDS
        rows, cols, n_bands = bands_data.shape
        n_samples = rows * cols

        # CREATE EMPTY ARRAY WITH SAME SIZE AS RASTER
        flat_pixels = bands_data.reshape((n_samples, n_bands))

        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

        # A list of colors for each class
        COLORS = [
            "#000000",  # 0 EMPTY
            "#00af11",  # 1 - Veg - Thick
            "#00e513",  # 2 - Veg - Sparse
            "#e9ff5a",  # 3 - Herbaceous
            "#f1ac34",  # 4 - Barren - Light
            "#a9852e",  # 5 - Barren - Dark
            "#2759ff",  # 6 - Water
            "#efefef",  # 7 - Roof - White
            "#d65133",  # 8 - Roof - Red
            "#cecece",  # 9 - Roof - Grey
            "#a0a0a0",  # 10 - Impervious - Light
            "#555555",  # 11 - Impervious - Dark
            "#000000"  # 12 - Shadows
        ]

        logger.debug("Classifing...")
        result = rf_classifier.predict(flat_pixels)

        # Reshape the result: split the labeled pixels into rows to create an image
        classification = result.reshape((rows, cols))

        # WRITE OUT THE CLASSIFIED ARRAY TO RASTER BASED ON PROPERTIES OF TRAINING RASTERS
        write_geotiff(loc_classified_file, classification, geo_transform, proj, classes, COLORS)
        logger.info("Classification created:", output_fname, " in ", str(datetime.now() - cl_start))
    else:
        print("The file exists. Skipping %s" % file)

    return loc_classified_file


def get_class_value(geom):
    """ TAKES A VARIABLE OF GEOMETRY TYPE AND RETURNS THE VALUE AT X,Y
    AS A PANDAS SERIES FOR FOR LOCAL RASTER 'CLASSRAS' """
    #print(row)
    #geom = row.geometry
    x = geom.centroid.x
    y = geom.centroid.y
    for val in classras.sample([(x, y)]):
        # print(np.ndarray.tolist(val))
        return pd.Series(val, index=[predicted_column])


def apply_and_concat(dataframe, field, func, column_names):
    return pd.concat((
        dataframe,
        dataframe[field].apply(
            lambda cell: pd.Series(func(cell), index=column_names))), axis=1)


def calculateGeom(row):
    #print(row)
    geom = row["geometry"]
    if row['PROJ'] == "NAD83 / UTM zone 11N":
        x = geom.centroid.x
        y = geom.centroid.y
        point = Point(transform(utm11, utm12, x, y))
        return point
    else:
        return geom


# LOCATIONS OF FOLDERS HOLDING ALL INPUT RASTER DATA
naip_dir = os.path.abspath(r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed")

base_datadir = os.path.abspath(r"M:\Data")
ndvi_dir = os.path.join(base_datadir, "NDVI")
savi_dir = os.path.join(base_datadir, "SAVI")
osavi_dir = os.path.join(base_datadir, "OSAVI")
msavi2_dir = os.path.join(base_datadir, "MSAVI2")
evi2_dir = os.path.join(base_datadir, "EVI2")
ndwi_dir = os.path.join(base_datadir, "NDWI")
ndsi_dir = os.path.join(base_datadir, "NDSI")

std3px_dir = os.path.join(base_datadir, "StdDev_3px")
std5px_dir = os.path.join(base_datadir, "StdDev_5px")
std10px_dir = os.path.join(base_datadir, "StdDev_10px")

base_landsatdir = os.path.join(base_datadir, "Landsat8")
landsat_dir = os.path.join(base_landsatdir, "byNAIPDOY_QQuads")

# LOCATION OF LANDSAT RASTER
landsat_file = os.path.os.path.join(base_landsatdir, "Landsat1to8_TOA_NAIPAcquiDate_merge_rectified.tif")

# LOCATION OF THE NDSI FILE
ndsi_file = os.path.join(ndsi_dir, "LandsatOLI_NDSI_30m.tif")

# LOCATION OF THE NDWI FILE
ndwi_file = os.path.join(ndwi_dir, "LandsatOLI_NDWI_30m.tif")

# LOCATION OF FILE CONTAINING CLASSIFICATION POINTS PRE-EXTRACT
loc_class_points = os.path.abspath(r"Q:\GoogleDrive\AridRiparianProject\WorkingDirectory\classificationPoints_join.shp")

# LOCATION OF FILE CONTAINING CLASSIFICATION POINTS POST EXTRACT
loc_points_wRaster_extracts = loc_class_points[:-4] + "_extracts.shp"

# DIRECTORY HOLDING VRTS BY QUARTER QUAD FOR VEGETATION INDICIES (FLOAT32) AND STD_DEV (UINT8))
qquad_vrt_dir = os.path.join(base_datadir, "QQuad_VRTs")

loc_classifiedQuarterQuads = os.path.join(base_datadir, "classifiedQuarterQuads")

# DEFINE THE PROJECTION USED OVER ARIZONA. USED FOR TRANSLATING POINT GEOMETRY LATER ON
utm11 = Proj(init="epsg:26911")
utm12 = Proj(init="epsg:26912")

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
            
rasters_names = naip + naip_vis + filters + landsat + landsat_vis
print(rasters_names)
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']



# IF VECTOR FILE OF POINTS WITH RASTER EXTRACTS DOESN'T EXIST, BUILD IT
if not os.path.exists(loc_points_wRaster_extracts):
    
    if "class_points" not in locals():
        print("READING IN %s as class_points" % loc_class_points)
        class_points = gpd.read_file(loc_class_points, crs={'init': 'epsg:26912'})
    
    if "utm_geom" not in class_points:
        print("ADDING COLUMN 'utm_geom' WITH CORRECT UTM COORDINATES FOR EACH QUARTER QUAD")
        # CREATE TRUE RASTER GEOMETRY COLUMN (BASED ON UTM)
        class_points["utm_geom"] = class_points.apply(calculateGeom, axis=1)
        
    if "NDSI" not in class_points:
        print("CREATING COLUMNS...")
        # CREATE EMPTY COLUMNS IN DATA FRAME FOR EACH RASTER VARIABLE
        for column in rasters_names:
            class_points[column] = np.NaN
    
    net_percentage = 0.0
    # ITERATE THROUGH DATAFRAME IN GROUPS BY NAIP_FILE. KEEPS US FROM OPENING/CLOSING RASTERS FOR EACH POINT - INSTEAD FOR EACH GROUP
    for loc_NAIPFile, group in class_points.groupby("NAIP_FILE"):
        print("\nStarting on %s" % loc_NAIPFile)
        loc_NAIPFile.replace("\\", "/")
    
        # LOOK FOR RASTERS FROM WHICH VALUES WILL BE EXTRACTED
        file = os.path.basename(loc_NAIPFile)
    
        vrt_naipvis = get_VegIndicies_VRT(file)
        #vrt_stddev = get_STDDev_VRT(file)
        gaussf_path = get_GaussianFile(file)
    
        landsat_path = createSubSetLandsat(loc_NAIPFile, landsat_file, landsat_dir).replace("\\","/")
    
        landsat_ndsi_path = createSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_dir).replace("\\", "/")
        landsat_ndwi_path = createSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_dir).replace("\\", "/")
    
        net_percentage += 100 * len(class_points.loc[class_points["NAIP_FILE"] == loc_NAIPFile])/len(class_points)
        print("Percentage of total: %d" % net_percentage)
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
                                    class_points.loc[class_points.NAIP_FILE == loc_NAIPFile, rasters_names] = \
                                        class_points.loc[class_points.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(get_values)
        
        print("Finished with group %s at %s" % (loc_NAIPFile, str(datetime.now())))
    
    # REMOVE ALL ROWS WHICH EXTRACTED NO DATA VALUES FROM LANDSAT
    #for column in landsat:
    #    class_points = class_points[class_points.loc[column] != 32766]
    
    print("Finished raster value extraction of %s points in %s" % (str(len(class_points)), str(datetime.now() - start)))

    # GEOPANDAS WON"T ALLOW MORE THAN ONE COLUMN WITH GEOMETRY TYPE. REMOVE THE utm_geom COLUMN CREATED PREVIOUSLY
    del class_points['utm_geom']
    print("COLUMNS:\n", class_points.columns)
    print("WRITING DATAFRAME TO OUTPUT...")
    class_points.to_file(loc_points_wRaster_extracts)

else:
    if "class_points" not in "locals":
        print("READING IN POINT FILE %s WITH RASTER EXTRACTS" % loc_points_wRaster_extracts)
        class_points = gpd.read_file(loc_points_wRaster_extracts)
        # WHEN WRITING TO SHAPEFILE, THE COLUMN NAMES ARE ABBREVIATED, GET NEW LIST OF NAMES
        rasters_names = class_points.columns.tolist()[18:-1]

class_points["utm_geom"] = class_points.apply(calculateGeom, axis=1)

# Split the points data frame into train and test
train, test = train_test_split(class_points, test_size=0.3)

# CREATE COLUMN FOR PREDICTED CLASSIFICATION VALUES
predicted_column = "CLASS_PREDICT"
test[predicted_column] = "Null"

#rasters_names = class_points.columns.tolist()[20:-2]
print("RASTER VARIABLES", rasters_names)

#rasters values used in random forest
temp_rasters = rasters_names[:]
rf_rasters = rasters_names[:]
for r in temp_rasters:
    #if "Landsat" in r:
    #if "StdDev_" in r:
        print(r)
    #    rf_rasters.remove(r)

print("Using %s" % rf_rasters)

overwrite = True

# TRAIN RANDOM FORESTS
rf_start = datetime.now()
print("Beginning Random Forest Train")
maxdepth = 35
#maxdepth = None
n_est = 15
n_job = 6
min_per_leaf = 10

rf = RandomForestClassifier(verbose=1, max_depth=maxdepth, n_estimators=n_est,
                            n_jobs=n_job, min_samples_leaf=min_per_leaf)

rf.fit(train[rf_rasters].dropna(),
       train[rf_rasters+["Class"]].dropna()["Class"])

print("Finished Fitting in", datetime.now() - rf_start)

# CREATE CLASSIFIED RASTERS FOR QUARTER QUADS USED IN TRAINING DATA FIRST
for loc_NAIPFile, group in class_points.groupby("NAIP_FILE"):
    classified_File = createClassifiedFile(loc_NAIPFile, rf)

    # EXTRACT PREDICTED PIXEL CLASSIFICATION TO TESTING DATAFRAME
    print("Extracting predicted classified values...")
    with rio.open(classified_File) as classras:
        # print(classras.indexes)
        test.loc[test.NAIP_FILE == loc_NAIPFile, [predicted_column]] = \
            test.loc[test.NAIP_FILE == loc_NAIPFile, "utm_geom"].apply(get_class_value)

# CREATE CONFUSION MATRIX, CLASSIFICATION REPORT, AND ACCURACY ASSESSMENT
print("Confusion matrix:\n%s" % str(metrics.confusion_matrix(test.Class, test.CLASS_PREDICT)))

target_names = ['Class %s' % s for s in classes]
print("Classification report:\n%s" % metrics.classification_report(test.Class, test.CLASS_PREDICT,
                                                                   target_names=target_names))

print("Classification accuracy: %f" % metrics.accuracy_score(test.Class, test.CLASS_PREDICT))

#We are going to observe the importance for each of the features and then store the Random Forest classifier using the joblib function of sklearn.

print(list(zip(rf_rasters, rf.feature_importances_)))
joblib.dump(rf, 'randomforestmodel_D' + str(maxdepth) + '_E' + str(n_est) + '.pkl')


# THEN CREATE CLASSIFIED RASTER FROM ALL OTHER QUARTER QUADS
for root, dirs, files in os.walk(naip_dir):
    for file in files:
        if file.endswith(".tif"):
            fpath = os.path.join(root,file)
            createClassifiedFile(fpath, rf)





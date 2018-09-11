import os
import geopandas as gpd
import pandas as pd
import RSAC_RegressionModel
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
import rasterio as rio
import numpy as np

logging.basicConfig(level=logging.INFO)

watersheds = os.path.abspath(r"M:/Data/inital_model_inputs/WBDHU4_Arizona.shp")
points = os.path.abspath(r"M:/Data/VBMPoints_2014.shp")
watersheds_dir = os.path.abspath(r"M:/Data/ValleyBottoms/Watersheds")

def getRasterNamesList(pdir):
    raster_paths = []
    raster_names = []
    for root, dirs, files in os.walk(pdir):
        for file in files:
            if file.endswith(".tif") or file.endswith(".img"):
                fpath = os.path.join(root, file).replace("\\", "/")
                if "elev_meters" not in file.lower():
                    raster_names.append(file[:-4])
                    raster_paths.append(fpath)
                else:
                    raster_names.insert(0, file[:-4])
                    raster_paths.insert(0, fpath)

    # elev_raster = os.path.join(pdir, "elev_meters.tif").replace("\\", "/")

    # raster_paths = [elev_raster] + float32_raster_paths
    # raster_names = ["elev_meters"] + raster_names
    #print("raster_names", raster_names)
    #print("raster_paths", raster_paths)
    return [raster_names, raster_paths]

points_info = RSAC_RegressionModel.extractValuesToVBPoints(watersheds, points, watersheds_dir)

points = points_info["points"]

points.to_file(r"M:\Data\VBMPoints_2014_extract.shp", driver="ESRI Shapefile")
#print(points.tail())

values_to_train_on = points_info["raster_names"]

formula = 'VB ~ ' + "+".join(values_to_train_on)
columns = ["VB"] + values_to_train_on
dta = points[columns].copy()

print("Formula: ", formula)
regressor = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
print(regressor.summary())
coefficients = regressor.fit().params
print("Predictors: ", coefficients)

intercept = coefficients.Intercept
coeffs = coefficients[1:]

print("Coefficients: ", coeffs)

predictors = getRasterNamesList(r"M:\Data\ValleyBottoms\Watersheds\1505\HRNHDPlusRasters1505\Predictors")[1]
print("Predictors: ", predictors)

calc = 0
for i in range(len(coefficients) - 1):
    print("Beginning on %s..." % predictors[i])
    with rio.open(predictors[i]) as raster:
        raster_array = raster.read(1).astype(float)

    print("\tCalculating on %s..." % predictors[i])
    calc += (float(coeffs[i]) * raster_array)

    del raster_array

calc += intercept
# string = string[:-4]
# string

outfile = r"M:\Data\ValleyBottoms\glm_test_1505.tif"
with rio.open(predictors[0]) as ras:
    kwargs = ras.profile

print("Calculating Valley Bottoms...\n")
#fp = eval(string)
lp = 1.0/(1.0 + np.exp(-1.0 * calc))

kwargs.update(dtype=np.float64)

print("Writing calculation to outfile %s...\n" % outfile)
with rio.open(outfile, 'w', **kwargs) as dst:
    dst.write_band(1, lp.astype(np.float64))
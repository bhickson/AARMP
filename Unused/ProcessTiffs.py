# SCRIPT RE-TILES AND COMPRESSES DIRECTORY OF RASTERS TO A NEW OUTPUT DIRECTORY

import gdal, os
from datetime import datetime
import time
from datetime import datetime

time.sleep(6500)

start = datetime.now()

print("Starting GDAL Translate at", start)

rasterDir = r"Q:\Arid Riparian Project\Data\az_1m_2015"
outDir = r"Q:\Arid Riparian Project\Data\NAIP_2015_Compressed"
count = 0

def getProj(file):
    raster = gdal.Open(file)
    return raster.GetProjectionRef()

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
                gdal_translate = "gdal_translate -overwrite -s_srs %s -t_srs %s -tr %s %s -tr %s %s %s %s" % (proj, ref_proj, resx, resy, width, height, srcfile, dstfile)
                print(file)
            ds = None



                #ds = gdal.Translate(opath, ds, creationOptions=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256",
                #                                                "BLOCKYSIZE=256", "NUM_THREADS=ALL_CPUS", "PREDICTOR=2"])
print("FINISHED")
print("ELAPSED", datetime.now() - start, ",  COUNT: ", count)
#average_time = round((datetime.now() - start).total_seconds() / count,2)
#print("AVERAGE TIME PER FILE: ", average_time)



import rasterio as rio
import numpy as np
import os, re
from datetime import datetime

precip_dir = r"M:\Data\PRISM\ppt"
base_file = r"M:\Data\PRISM\ppt\PRISM_ppt_stable_4kmD2_20150501_bil.bil"
limit = 5
clip_file = r"M:\Data\NAIP_AZ_Boundary.gpkg"

def agg_vals(precip_array, base_array, doy_vals, day):
    larger_than_limit = precip_array >= limit # will return 1 for all new precip values
    base_unset = base_array == 0 # returns 1 for all base array values still equal to zero
    new_values = larger_than_limit * base_unset * precip_array
    new_doy = larger_than_limit * base_unset * int(day)

    base_array += new_values
    doy_vals += new_doy

    return [base_array,doy_vals]

def getDayOfYear(file):
    # PRISM DOWNLOADS ARE IN FORMAT PRISM_ppt_stable_4kmD2_20150501_bil.bil
    day_string = re.findall(r'\d+', file)[2]  #slips string in to list of numeric chunks (e.g. [4, 2, 20150501])
    dateobj = datetime.strptime(day_string, '%Y%m%d')
    doy = str(dateobj.timetuple().tm_yday)
    return doy

def clipRasterToAZ(clip, inras):
    outras = inras[:-4] + "_AZ.tif"
    cmd = 'gdalwarp -q -cutline %s -crop_to_cutline %s %s' % (clip, inras, outras)
    os.system(cmd)



with rio.open(base_file) as raster:
    kwargs = raster.meta
    base_raster = np.zeros(raster.shape, dtype=rio.float32)
    doy_raster = np.zeros(raster.shape, dtype=rio.float32)

    for root, dirs, files in os.walk(precip_dir):
        for file in files:
            if file.endswith(".bil"):
                print("STARTING FILE ", file)
                day = getDayOfYear(file)
                fpath = os.path.join(root,file)
                with rio.open(fpath) as raster:
                    precip = raster.read(1).astype(float)
                    rasters = agg_vals(precip, base_raster, doy_raster, day)
                    base_raster = rasters[0]
                    doy_raster = rasters[1]

    kwargs.update(
        dtype=rio.float32,
        count=1,
        compress='lzw')

    ofile_firstprecip = r"M:\Data\PRISM\FirstPrecipAmount_min" + str(limit) + "mm.tif"

    with rio.open(ofile_firstprecip, 'w', **kwargs) as dst:
        dst.write_band(1, base_raster.astype(rio.float32))


    kwargs.update(
        dtype=rio.uint16)

    ofile_doy = r"M:\Data\PRISM\FirstDOYPrecip_min" + str(limit) + "mm.tif"
    with rio.open(ofile_doy, 'w', **kwargs) as dst:
        dst.write_band(1, doy_raster.astype(rio.uint16))

    # CLIP THE RASTER PRODUCTS TO OUR AOI (Arizona)
    clipRasterToAZ(clip_file, ofile_firstprecip)
    clipRasterToAZ(clip_file, ofile_doy)
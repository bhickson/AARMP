import os
from scipy import ndimage
import numpy as np
import rasterio as rio
import Utilities as utilities
from scipy import ndimage as ndi
from datetime import datetime

def vegIndexCalc(file, out_dir, indicies):
    #naip_file, out_dir, [type])
    """Takes an input naip file, a parent output directory, and a list of indicies and creates an output derivative
     vegetation index from the input naip bands written to a folder of the of the name of the index in the output dir"""

    # GET FILE NAME FROM PATH
    fname = os.path.basename(file)

    # Read raster bands directly to Numpy arrays.
    with rio.open(file) as raster:
        bandBlue = raster.read(3).astype(float)
        bandRed = raster.read(1).astype(float)
        bandNIR = raster.read(4).astype(float)
        kwargs = raster.meta
        vi = np.zeros(raster.shape, dtype=rio.float32)

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    for veg_index in indicies:
        # CREATE NEW DIRECTORY FOR VI TYPE
        outdir = os.path.join(out_dir, veg_index)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        ofile = outdir + "\\" + veg_index + "_" + fname
        if veg_index == "NDVI":
            vi = (bandNIR - bandRed) / (bandNIR + bandRed)
        else:
            # IMPORTANT: Because of the soil line value in the SAVI indicies,
            #  all band value must be normalized between 0 and 1.
            bandRed /= 255.0
            bandNIR /= 255.0
            bandBlue /= 255.0
            if veg_index == "SAVI":
                l = 0.5
                vi = ((bandNIR - bandRed) / (bandNIR + bandRed + l)) * (1 + l)
            elif veg_index == "MSAVI2":
                vi = ((2 * bandNIR + 1) - np.sqrt(np.square(2 * bandNIR + 1) - (8 * (bandNIR - bandRed)))) / 2
            elif veg_index == "EVI2":
                g = 2.5      # gain factor
                l = 1.0        # soil adjustment factor
                c_one = 6.0    # coefficient
                c_two = 7.5  # coefficient
                vi = 2.5 * ((bandNIR - bandRed) / (bandNIR + (2.4 * bandRed) + 1))
            elif veg_index == "OSAVI":
                vi = (bandNIR - bandRed) / (bandNIR + bandRed + 0.16)

        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')

        with rio.open(ofile, 'w', **kwargs) as dst:
            dst.write_band(1, vi.astype(rio.float32))


def standardDeviation(naip_file, oloc, window_size=5, overwrite=False):
    raster_paths = []
    # GET FILE NAME FROM PATH
    fname = os.path.basename(naip_file)
    window_dir = os.path.join(oloc, "StdDev_" + str(window_size) + "px")
    utilities.useDirectory(window_dir)

    # CHECK IF OUTPUT HAS ALREADY BEEN  MADE
    for i in range(4):
        band_num = str(i+1)

        # CREATE NEW DIRECTORY FOR BAND TYPE
        band_dir = os.path.join(window_dir, "band" + band_num)
        utilities.useDirectory(band_dir)

        out_file = os.path.join(band_dir, "stddev_" + fname[:-4] + "band" + str(band_num) + ".tif")

        raster_paths.append(out_file)

        if not overwrite and os.path.exists(out_file):
            #print("SKIPPING %s...", % out_file)
            continue
        else:
            if "bands" not in locals():
                with rio.open(naip_file) as raster:
                    bandBlue = raster.read(1).astype(float)
                    bandGreen = raster.read(2).astype(float)
                    bandRed = raster.read(3).astype(float)
                    bandNIR = raster.read(4).astype(float)
                    kwargs = raster.profile

                bands = [bandBlue, bandGreen, bandRed, bandNIR]

            band = bands[i]
            # https://stackoverflow.com/a/33497963
            std_dev = np.zeros(raster.shape, dtype=rio.uint8)
            win_mean = ndimage.uniform_filter(band, window_size)
            win_sqr_mean = ndimage.uniform_filter(band ** 2, window_size)
            std_dev = np.sqrt(win_sqr_mean - win_mean**2)

            kwargs.update(
                dtype=rio.uint8,
                count=1,
                compress='lzw')

            ofile = os.path.join(band_dir, "stddev_" + fname[:-4] + "band" + band_num + ".tif")
            with rio.open(ofile, 'w', **kwargs) as dst:
                dst.write_band(1, std_dev.astype(rio.uint8))

    return raster_paths


def gaussianCalc(naip_file, oloc, sigma=1, overwrite=False):
    gc_start = datetime.now()
    raster_paths = []
    # GET FILE NAME FROM PATH
    fname = os.path.basename(naip_file)
    sigma_dir = os.path.join(oloc, "Gauss_" + str(sigma))
    utilities.useDirectory(sigma_dir)

    out_file = os.path.join(sigma_dir, "gauss_" + str(sigma) + "_" + fname[:-4] + ".tif")

    if not os.path.exists(out_file) or overwrite:
        #print("\t\tStarting Gaussian Calc for %s" % naip_file)
        with rio.open(naip_file) as naip_ras:
            #print(naip_ras)
            kwargs = naip_ras.profile
            naip_array = naip_ras.read().astype(rio.float32)

        kwargs.update(
            dtype=rio.float32
        )

        gaussian_array = ndi.gaussian_filter(naip_array, 1)

        with rio.open(out_file, 'w', **kwargs) as dst:
            dst.write(gaussian_array.astype(rio.float32))
            
        print("\t\tFinished gauss calc in %s" % (str(datetime.now()-gc_start)))
    #else:
    #    print("SKIPPING %s..." % out_file)

    return out_file


def calculateLandsatIndicies(landsat_stack, ndwi_outdir=None, ndsi_outdir=None, overwrite=False):
    """ Given the input landsat file, calculates
    1. Normalized difference wetness/water index:
        NDWI = 3-5/3+5 - green - nir / green + nir
    2. Normalized difference soil index:
        NDSI = 6-5/6+5 - tm bands 5-4/5+4 (SWIR 1.5 to 1.7) - NIR
    """

    with rio.open(landsat_stack) as raster:
        print("Starting NIR band read... %s" % str(datetime.now()))
        bandNIR = raster.read(5).astype(rio.float32)
        no_data = bandNIR == 32766
        kwargs = raster.meta

        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw',
            driver='GTiff',
            nodata=-9999
        )

        if ndwi_outdir:
            ndwi_ofile = os.path.join(ndwi_outdir, "LandsatOLI_NDWI_30m.tif")

            if not os.path.exists(ndwi_outdir) or overwrite:
                print("Starting green band read... %s" % str(datetime.now()))
                bandGreen = raster.read(3).astype(rio.float32)


                print("\nStarting NDWI Calc... %s" % str(datetime.now()))
                ndwi = (bandGreen - bandNIR) / (bandGreen + bandNIR)
                ndwi[no_data] = -9999

                with rio.open(ndwi_ofile, 'w', **kwargs) as dst:
                    dst.write_band(1, ndwi.astype(rio.float32))
            else:
                print("NDWI File already exists. Skipping calculation")

        if ndsi_outdir:
            if not os.path.exists(ndwi_outdir) or overwrite:
                print("Starting SWIR band read... %s" % str(datetime.now()))
                bandSWIR1 = raster.read(6).astype(rio.float32)

                print("Starting NDSI Calc... %s" % str(datetime.now()))
                ndsi = (bandSWIR1 - bandNIR) / (bandSWIR1 + bandNIR)
                ndsi[no_data] = -9999

                ndsi_ofile = os.path.join(ndsi_outdir, "LandsatOLI_NDSI_30m.tif")

                with rio.open(ndsi_ofile, 'w', **kwargs) as dst:
                    dst.write_band(1, ndsi.astype(rio.float32))
            else:
                print("NDSI File already exists. Skipping calculation")

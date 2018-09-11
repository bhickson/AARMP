import glob, os, math
import rasterio as rio
from datetime import datetime
import shutil
import sys
import Utilities as utils
import numpy as np
import gdal, osr
import geopandas as gpd
from gdalconst import GA_ReadOnly

from getLandsatData import rasterizeVector

def createAcquiRasters(acqui_day_raster, acqui_day_opath, ls_no_data):

    with rio.open(acqui_day_raster) as ras:
        raster_array = ras.read().astype(rio.int16)
        unique = np.unique(raster_array)
        print("UNIQUE VALUES/DAYS: ", unique)
        kwargs = ras.profile

        kwargs.update(
            nodata=ls_no_data,
            dtype=rio.int16,
            crs="+proj=utm +zone=12 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
        )

        print(kwargs)
        # Iterate through raster for unique values, which are the julian days of the year (DOY)
        for val in unique:
            # Only iterate valid DOYs
            if val != 0 and val != 32766:
                print(val)
                nodata = raster_array == 32766  # value outside of state boundaries
                # Set pixels with DOY value to landsat no data value so that they will later
                # know to be filled in by landsat values
                val_ras = np.where(raster_array == val, ls_no_data, 32764)
                val_in_state = np.where(raster_array != 32766, val_ras, raster_array)

                ofile = os.path.join(acqui_day_opath, "day_" + str(val) + ".tif")
                if not os.path.exists(ofile):
                    print("Writing output file", ofile)
                    with rio.open(ofile, 'w', **kwargs) as dst:
                        dst.write(val_in_state.astype(rio.int16))


def createDictofLandsatDays(landsat_days_dir):
    landsat_dict = {}
    for file in os.listdir(landsat_days_dir):
        if file.endswith(".tif"):
            day = int(file.split("_")[1])
            fpath = os.path.join(landsat_days_dir, file)
            landsat_dict[day] = fpath

    return landsat_dict


def rasterReplace(acqui_array, landsat_array):
    # SET NO DATA VALUES IN TH LANDSAT ARRAY TO THE VALID DAY PLACEHOLDER VALUE (32765) OF THE NAIP DATA. THIS IS WHAT WE TEST AGAINST TO SEE IF LANDSAT VALUES HAVE BEEN FILLED FOR THAT DATE
    #landsat_sel = np.where(landsat_array == 32767, 32765, landsat_array)

    # WHERE THE ACQUISITION DATE ARRAY IS EQUAL TO THE VALID VALUE PLACEHOLDER (32765), REPACE WITH LANDSAT VALUES, OTHERWISE, KEEP SAME VALUES
    ras = np.where(acqui_array == 32767, landsat_array, acqui_array)

    return ras


def coincidenceTest(acqui_array, landsat_array, ls_no_data):
    """ LOOKS AT AN ARRAY OF LANDSAT DATA COLLECTED ON A SINGLE DATA AND AN ARRAY CONTAINING THE NAIP ACQUISITION DATE.
    FINDS THE RELEVANT (NOT NO DATA) LANDSAT VALUES AND THE RELEVANT (NOT NO DATA AND NOT OUT OF STATE) NAIP VALUES AND
     OVERLAYS THEM TO FIND WHICH INDICIES (I.E. PIXELS) ARE CONINCIDENT AND VALID (1).
     THE LANDSAT NO DATA VALUE IS 32767
     THE NAIP NO DATA VALUE IS 32767
     THE NAIP OUT OF STATE VALUE IS 32766"""
    landsat_test = landsat_array != ls_no_data
    #print(landsat_test)
    acqui_test_naipdate = acqui_array[0] != 32764
    acqui_test_instate = acqui_array != 32766
    #print(acqui_test)
    # THREE ARRAYS OF BINARY VALUES (True, False). AT INDICIES WHERE ARE ALL TRUE THEY EVALUATE TO 1.
    test = landsat_test * acqui_test_naipdate * acqui_test_instate
    if 1 in test:
        return True
    else:
        return False


def getClosestDay(day, ldict):
    """Find the closest key value in landsat_days dictionary for day passed to function"""
    landsat_day = min(ldict, key=lambda x: abs(x - day))

    return landsat_day


def writeLandsatToAcquiDay(naip_acquiDir, days_dir, outDir, ls_no_data):
    """ Takes Landsat scenes which have been clipped to NAIP day"""

    print("%s - Write Landsat data by date to NAIP acquisition day" % (str(datetime.now())))
    tifs = [f for f in glob.iglob(naip_acquiDir + "/*.tif")]  # GET ALL TIFS IN THE NAIP ACQUISITION DATE DIRECTORY

    #with rio.open(r"M:\Data\Landsat8\Landsat1to8_TOA_NAIPAcquiDate_merge_fix.tif") as former:
    #    new_ras = former.read().astype(rio.uint16)

    for tif in tifs:
        start = datetime.now()
        landsat_days = createDictofLandsatDays(days_dir)
        file = os.path.basename(tif)
        ofile = os.path.join(outDir, file)

        coincident_count = 0

        if os.path.exists(ofile):
            print("%s file exists, skipping..." % ofile)
        else:
            day = int(file.split("_")[1].split(".")[0])

            # get closest landsat acquisition day for naip acquisition day
            landsat_day = getClosestDay(day, landsat_days)

            print("Found Landsat day %s for NAIP day %s" % (landsat_day, day))

            # OPEN AND READ THAT NAIP ACQUISITION DAY FILE
            with rio.open(tif) as ar:
                acqui_ras = ar.read().astype(rio.uint16)
                kwargs = ar.profile

                kwargs.update(
                    count=8,
                    crs="+proj=utm +zone=12 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
                )

                new_ras = acqui_ras
                iterations = 0
                print("Starting iterations for ", landsat_day)
                invalid = True
                while invalid:
                    iterations += 1
                    if iterations != 1:
                        print(iterations, "Still have bad values. Iterating on", landsat_day)
                    landsat_ras = landsat_days[landsat_day]
                    # TEST TO SEE IF THERE IS OVERLAP BETWEEN THE VALID (NOT NO DATA) LANDSAT VALUES AND THE NAIP DAY VALUES
                    with rio.open(landsat_ras) as lr:
                        # NO NEED TO READ IN ALL LANDSAT BANDS FOR THIS TEST (TAKES LONG)
                        landsat_band = lr.read(1).astype(rio.int16)
                        if coincidenceTest(new_ras, landsat_band, ls_no_data):
                            coincident_count += 1
                            print("Acquisition date coverage and landsat acquisition values coincident for %s" % (os.path.basename(landsat_ras)))
                            raster_array = lr.read().astype(rio.int16)
                            new_ras = rasterReplace(new_ras, raster_array)
                        #else:
                        #    print("Acquisition date and landsat values at date are not coincident for landsat %s" % (os.path.basename(landsat_ras)))

                    raster_array = None
                    landsat_band = None

                    # remove landsat day from dictionary to prevent re-associating
                    landsat_days.pop(landsat_day, None)
                    if len(landsat_days) > 0 or coincident_count == 10:
                        landsat_day = getClosestDay(day, landsat_days)
                    else:
                        print("No more landsat days to iterate. Writing output")
                        break
                    if (ls_no_data not in new_ras):  # or  (32765 in new_ras):   # 32765 are valid NAIP acquisition day values and 32767 are nodata landsat values
                        invalid = False
                        print("No Landsat nodata values %d left in date %s coverage. Writing output" % (ls_no_data, day))



                print("Writing output file", ofile, "with kwargs: \n", kwargs)
                with rio.open(ofile, 'w', **kwargs) as dst:
                    dst.write(new_ras.astype(rio.int16))

            print("Finished with", day, "in", datetime.now() - start), "\n"


def mergeDirectoryToFile(dir, out_file_path):
    """Takes an input directory and an output file to create. Merges all tif files from directory into the ouput file.
    If output file already exists, prints out notice and does not overwrite"""

    if not os.path.exists(out_file_path):
        raster_files = []
        for file in os.listdir(dir):
            if file.lower().endswith(".tif"):
                fpath = os.path.join(dir, file)
                raster_files.append(fpath)

        input_files = " ".join(raster_files)

        #with rio.open(raster_files[0]) as raster:
        #    no_data_val = raster.profile['nodata']

        in_nodata_val = 32764
        out_nodata_val = 32766

        print("%s - Starting file merge of directory %s to file %s" % (str(datetime.now()), dir, out_file_path))

        gdal_merge_dir = "python gdal_merge.py -n %s -a_nodata %s -o %s %s" % (str(in_nodata_val), str(out_nodata_val), out_file_path, input_files)
        os.system(gdal_merge_dir)
    else:
        print("Output file %s already exists." % out_file_path)

    print("%s - Landsat merge finished" % str(datetime.now()))


def createVRT(indir, outfilename):
    outfile = os.path.join(indir, outfilename)

    infiles = glob.glob(indir + "/" + "*.tif")

    build_vrt = "gdalbuildvrt %s %s" % (outfile, " ".join(infiles))
    print("Building VRT...")
    os.system(build_vrt)
    print("\t...Finished")

    return outfile


def getDissolvedPathShapefile(landsat_path):
    paths_dir = os.path.join(landsat_dir, "dissolvedPaths")
    for f in os.listdir(paths_dir):
        if landsat_path in f and f.endswith(".shp"):
            fp = os.path.join(paths_dir, f)
            return fp


def clipDOYFiles(in_dir, out_dir, naip_acqui_ras):
    # print(os.listdir(dir))

    temp_folder = "temp_resized"
    temp_dir = os.path.join(out_dir, temp_folder)
    print(temp_dir)
    utils.useDirectory(temp_dir)

    tifs = [f for f in glob.iglob(in_dir + "/*.tif")]
    print(tifs)
    for file in tifs:
        count = 0

        fname = os.path.splitext(file)[0]
        print(fname)

        temp_ras = os.path.join(temp_dir, os.path.basename(fname) + "_clipped_resized.tif")
        trans_ras = os.path.join(out_dir, os.path.basename(fname) + "_clipped.tif")

        if not os.path.exists(trans_ras):
            count = 0
            count += 1
            print("\n", file)
            landsat_path = int(fname[-2:])
            vector = getDissolvedPathShapefile(str(landsat_path))

            # CLIP MERGE RASTER PATH TO REMOVE EDGES WHICH HAVE UNMASKED CLOUDS.
            # This happens because the the fmask alg looks at the direction of
            # clouds to determine shadows and edges don't get masked
            print("%d - Starting clip of %s to %s" % (count, file, temp_ras))
            gdal_clip = "gdalwarp -overwrite -srcnodata 32767 -dstnodata 32767 -tr 30 30 -crop_to_cutline" \
                        " -cutline %s %s %s" % (vector, file, temp_ras)
            os.system(gdal_clip)

            data = gdal.Open(naip_acqui_ras, GA_ReadOnly)
            projwin_srs = data.GetProjection()

            geoTransform = data.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * data.RasterXSize
            miny = maxy + geoTransform[5] * data.RasterYSize

            # NEED TO RESIZE RASTER TO FIT WHOLE EXTENT OF AOI
            print("\tStarting Translate of ", temp_ras)
            projwin = str(minx) + " " + str(maxy) + " " + str(maxx) + " " + str(miny)
            gdaltranslate = 'gdal_translate -projwin ' + projwin + " -projwin_srs " + projwin_srs + " -of GTiff " + temp_ras + " " + trans_ras
            os.system(gdaltranslate)

            os.remove(temp_ras)


def mergeDOYFiles(in_dir, out_dir):
    for dir in os.listdir(in_dir):
        dir_path = os.path.join(in_dir, dir)
        raster_files = []
        for file in os.listdir(dir_path):
            if file.upper().endswith(".TIF"):
                fpath = os.path.join(dir_path, file)
                raster_files.append(fpath)
            path = int(file[11:14])

        # GDAL MERGE WILL WRITE THE 'the last image will be copied over earlier ones' so sort list of images so
        # that the image with the lowest cloud cover is written last
        raster_files = sorted(raster_files, key=str.lower, reverse=True)
        input_files = " ".join(raster_files)
        ofile_name = "toaMask_" + dir + "_path" + str(path) + ".tif"

        output_file = os.path.join(out_dir, ofile_name)
        if os.path.exists(output_file):
            print("%s already exists. Skipping..." % (output_file))
        else:
            print("Starting file merge of directory", dir_path, "to file", output_file)
            gdal_merge_dir = "python gdal_merge.py -tap -a_nodata 32767 -o %s %s" % (output_file, input_files)
            os.system(gdal_merge_dir)


def createLandatDOYCompositeForAOI(naip_acquidoy_raster, data_dir, ls_toa_dir, ls_dir,
                                   ls_toa_naipdate_merge, ls_nodata=32767):
    """ Master landsat creator function. Takes a directory of landsat TOA tiles by doy (see getLandsatData.py) and then.
        1. Takes a raster of the NAIP acquisition days (julian day of year) and creates new raster for each unique day
            value
        2. Merges all the landsat tiles who share a same-day of collection (typically the whole path on that day)
        3. Clips the merged landsat files from #2 to remove unmasked cloud edges
        4. Looks at each naip doy area and trys to replace with the values from the closest landsat collection day at
            that location. Results in 1 file for each naip acquisition date
        5. Merges all landsat files from #4 backinto 1 single landsat composite.
    """
    # Create directory structure
    merged_dir = os.path.join(ls_dir, "TOA_Corrected_PathMerge")
    utils.useDirectory(merged_dir)
    clipped_dir = os.path.join(ls_dir, "TOA_Corrected_PathMerge_Clipped")
    utils.useDirectory(clipped_dir)
    landsatBy_NAIPAcqui_dir = os.path.join(ls_dir, "acqui_clips")
    utils.useDirectory(landsatBy_NAIPAcqui_dir)
    # LOCATION OF RASTERS FOR EACH INDIVIDUAL NAIP DOY
    naip_acquidoy_dir = os.path.join(data_dir, "acquisitionDate_rasters")
    utils.useDirectory(naip_acquidoy_dir)

    createAcquiRasters(naip_acquidoy_raster, naip_acquidoy_dir, ls_nodata)

    mergeDOYFiles(ls_toa_dir, merged_dir)

    clipDOYFiles(merged_dir, clipped_dir, naip_acquidoy_raster)

    writeLandsatToAcquiDay(naip_acquidoy_dir, clipped_dir, landsatBy_NAIPAcqui_dir, ls_nodata)

    mergeDirectoryToFile(landsatBy_NAIPAcqui_dir, ls_toa_naipdate_merge)



if __name__ == "__main__":
    landsat_nodata = 32767

    # SET ALL DIRECTORIES
    dataDir = os.path.abspath(r"M:/Data")

    landsat_dir = os.path.abspath(r"M:\Data\Landsat8")

    LT1_dir = os.path.join(landsat_dir, "LT1")
    utils.useDirectory(LT1_dir)
    toaDir = os.path.join(landsat_dir, "TOA_Corrected")
    utils.useDirectory(toaDir)

    # LOCATION OF NAIP ACQUISITION DATE RASTER
    acqui_vector = os.path.abspath(r"Q:\Arid Riparian Project\Data\NAIP2015_AcquisitionDates_Dissolve.gpkg")
    acqui_raster = os.path.abspath(r"M:\Data\NAIP2015_AcquisitionDates_Dissolve.tif")
    rasterizeVector(acqui_vector, acqui_raster)

    # FINAL OUT FILE FROM THIS SCRIPT
    landsat_toa_naipdate_merge = os.path.join(landsat_dir, "Landsat1to8_TOA_NAIPAcquiDate_merge.tif")

    createLandatDOYCompositeForAOI(acqui_raster, dataDir, toaDir, landsat_dir,
                                   landsat_toa_naipdate_merge, ls_nodata=landsat_nodata)


"""
USED ONLY TO FIX BAD values FROM writeLandsatToAcquiDay RUN...
with rio.open(landsat_toa_naipdate_merge) as raster:
    print("STARTING FIX")
    ofile = landsat_toa_naipdate_merge = os.path.abspath(r"M:\Data\Landsat8\Landsat1to8_TOA_NAIPAcquiDate_merge_fix.tif")

    allbands = raster.read().astype(rio.uint16)
    kwargs = raster.profile

    print("SETTING VALUES")
    val_ras = np.where(allbands[7] == 0, 32764, allbands)

    print("WRITING FILE")
    with rio.open(ofile, 'w', **kwargs) as dst:
        dst.write(val_ras.astype(rio.int16))
"""
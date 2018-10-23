import os
import glob
import math
import requests

import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import Utilities as utilities
import shutil
import gdal
from gdalconst import GA_ReadOnly


from datetime import datetime
from bs4 import BeautifulSoup

try:
    #  https://bitbucket.org/chchrsc/python-fmask/downloads/
    import fmask
except:
    print("Package fmask not found. Installing...\n")
    utilities.installPackage("fmask")

try:
    #  https://bitbucket.org/chchrsc/rios/downloads/
    import rios
except:
    print("Package rios not found. Installing...\n")
    utilities.installPackage("rios")

# GDAL has some trouble finding espg reference files. make sure it knows
gdal.SetConfigOption('PROJSO', r"C:\Program Files\Anaconda3\Library\bin\proj.dll")


# LOCATE fmask USGS SCRIPTS
for root, dirs, files in os.walk(os.path.abspath("./python-fmask-0.4.5")):
    for file in files:
        if file == "fmask_usgsLandsatSaturationMask.py":
            fpath_satMask = os.path.join(root,file)
        elif file == "fmask_usgsLandsatMakeAnglesImage.py":
            fpath_angles = os.path.join(root,file)
        elif file == "fmask_usgsLandsatTOA.py":
            fpath_toa = os.path.join(root,file)
        elif file == "fmask_usgsLandsatStacked.py":
            fpath_stack = os.path.join(root,file)


def doyToDate(doy, year):
    date = datetime.strptime(str(year) + "+" + str(int(doy)), "%Y+%j")
    date = date.strftime("%Y-%m-%d")
    return date


def downloadList(frame, downloadDir):
    total_scenes = len(frame)
    count = 0
    entity_list = {}
    for i, row in frame.iterrows():
        entity_list[row.entityId] = row.date.strftime("%Y-%m-%d")
        count += 1
        print(count, " of ", total_scenes)
        # Print some the product ID
        #print('\tChecking content: ')
        #row.geometry = row.geometry.buffer(-0.05)

        # Request the html text of the download_url from the amazon server.
        # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
        response = requests.get(row.download_url)

        # If the response status code is fine (200)
        if response.status_code == 200:
            # Import the html to beautiful soup
            html = BeautifulSoup(response.content, 'html.parser')

            # Create the dir where we will put this image files.
            date_dir = os.path.join(downloadDir, row.date.strftime("%Y-%m-%d"))
            utilities.useDirectory(date_dir)

            entity_dir = os.path.join(date_dir, row.entityId)
            utilities.useDirectory(entity_dir)

            # Second loop: for each band of this image that we find using the html <li> tag
            for li in html.find_all("li"):
                # Get the href tag
                file = li.find_next("a").get("href")
                opath = os.path.join(entity_dir, file)

                if not file.endswith(".ovr") and not os.path.exists(opath):   # don't download overview files, skip existing files
                    print('\tDownloading: {}'.format(file))

                    # Download the files
                    # code from: https://stackoverflow.com/a/18043472/5361345

                    response = requests.get(row.download_url.replace("index.html", file))

                    with open(opath, 'wb') as f:
                        for n, chunk in enumerate(response.iter_content(chunk_size=512), start=1):
                            f.write(chunk)

                    del response


def parseMTLFile(file):
    with open(file, 'r') as f:
        contents = f.read().replace(" ", "").split("\n")

    mtl_dict = {}
    for line in contents:
        parsed = line.split("=")
        if len(parsed) == 2:
            # print(line)
            mtl_dict[parsed[0]] = parsed[1]

    return mtl_dict


def calculateRasterTOA(mtl_file, toa_dir):
    """ RASTER TOA CALCULATION. USES VALUES IN MTL FILE. RETURNS RASTER OBJECT """

    mtl_values = parseMTLFile(mtl_file)

    doy = os.path.basename(mtl_file)[13:16]

    parent_dir = os.path.abspath(os.path.join(mtl_file, os.pardir))  # GET PARENT DIRECTORY OF MTL

    os.makedirs(toa_dir, exist_ok=True)

    doy_dir = os.path.join(toa_dir, doy)
    os.makedirs(doy_dir, exist_ok=True)

    tifs = [f for f in glob.iglob(parent_dir + "/*.TIF")]  # GET ALL TIFS IN THE MTLs DIRECTORY
    for tif in tifs:
        file = os.path.basename(tif)  # GETTING ONLY FILE NAME
        ofile = os.path.join(doy_dir, file)
        if not os.path.exists(ofile):

            if "BQA" not in file:
                band = tif.split("_")[1].split(".")[0].replace("B", "")  # GET LANDSAT BAND
                if band != "10" and band != "11":
                    #  ρλ' = Mρ* Qcal + Aρ
                    #  ρλ' = TOA planetary reflectance, without correction for solar angle. Note that ρλ' does not contain a
                    #       correction for the sun angle.
                    #  Mρ = Band-specific multiplicative rescaling factor from the metadata (Reflectance_Mult_Band_x, where x
                    #       is the band number)
                    #  Aρ = Band-specific additive rescaling factor from the metadata (Reflectance_Add_Band_x, where x is the
                    #       band number)
                    #  Qcal = Quantized and calibrated standard product pixel values (DN).
                    #  θSE = Local sun elevation angle.The scene center sun elevation angle in degrees is provided in the
                    #       metadata(Sun Elevation).

                    mp = float(mtl_values["REFLECTANCE_MULT_BAND_" + band])
                    ap = float(mtl_values["REFLECTANCE_ADD_BAND_" + band])
                    se = float(mtl_values["SUN_ELEVATION"])

                    with rio.open(tif) as raster:
                        qcal = raster.read(1).astype(rio.float32)
                        kwargs = raster.meta

                        toa_reflectance = (mp * qcal) + ap / math.sin(se)  # Top of atmosphere reflectance corrected for sun angle

                        kwargs.update(
                            dtype=rio.float32,
                            count=1,
                            compress='lzw')

                        with rio.open(ofile, 'w', **kwargs) as dst:
                            dst.write_band(1, toa_reflectance)


def findAndReproject(folder_path, target_EPSG="32612"):
    """ crawls directory and looks for files whose projection doesn't match target_EPSG value. Then reprojects file
     and replaces original """
    repro_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.upper().endswith(".TIF"):
                fpath = os.path.join(root, file)

                with rio.open(fpath) as src:
                    epsg = src.crs['init'].split(":")[1]

                if epsg != target_EPSG:
                    repro_count += 1
                    print("Reprojecting %s ..." % fpath)
                    new_raster = os.path.basename(file)[:-4] + "_" + target_EPSG + ".tif"
                    print(new_raster)
                    gdal_warp_command = "gdalwarp -r near -tap -tr 30 30 -t_srs EPSG:%s %s %s" % (target_EPSG, fpath, new_raster)
                    os.system(gdal_warp_command)
                    os.remove(fpath)
                    shutil.move(new_raster, fpath)

    print(repro_count, "total files reprojected")


def buildIfNotExist(calc, dir, file, command_line_arg):
    if not os.path.exists(file):
        print("\nBeginning %s image calc for %s" % (calc, dir))
        print(command_line_arg)
        os.system(command_line_arg)


def createCloudMaskedTOA(in_dir, out_dir):
    """ uses the fmask algorithm to create the Top of Atmosphere interpretations of landsat scenes and a mask of
    clouds and cloud shadows. Ultimately, creates new output landsat scenes of TOA masked by clouds"""
    scene_count = 0
    for root, dirs, files in os.walk(in_dir):
        for dir in dirs:
            if dir.startswith("LC8"):
                scene_count += 1
                print(scene_count)
                doy = os.path.basename(dir)[13:16]

                doy_dir = os.path.join(out_dir, doy)
                os.makedirs(doy_dir, exist_ok=True)

                landsat_scene_dir = os.path.join(root, dir)

                #ofile = os.path.join(doy_dir, dir + ".tif")
                mtl_file = [f for f in glob.iglob(landsat_scene_dir + "/*_MTL.txt")][0]  # GET ALL REFLECTANCE TIFS IN THE MTLs DIRECTORY
                cloudcover = parseMTLFile(mtl_file)["CLOUD_COVER"]
                # make sure that there are two character spaces. e.g. 1.13 should be 01.13
                if len(cloudcover.split(".")[0]) == 1:
                    cloudcover = "0" + cloudcover
                ofile = "CC" + str(cloudcover) + "_" + dir + ".tif"
                ofile_path = os.path.join(doy_dir, ofile)


                if not os.path.exists(ofile_path):
                    # Create raster stacks of thermal and reflective bands separately
                    reflective_bands = [f for f in glob.iglob(
                        landsat_scene_dir + "/LC8*_B[1-7,9].TIF")]  # GET ALL REFLECTANCE TIFS IN THE MTLs DIRECTORY
                    thermal_bands = [f for f in glob.iglob(
                        landsat_scene_dir + "/LC8*_B1[0,1].TIF")]  # GET ALL REFLECTANCE TIFS IN THE MTLs DIRECTORYreflective_bands)
                    mtl_file = [f for f in glob.iglob(landsat_scene_dir + "/*_MTL.txt")][
                        0]  # GET ALL REFLECTANCE TIFS IN THE MTLs DIRECTORY

                    sep = os.path.sep

                    reflectance_image = "%s%sRefl_%s.tif" % (landsat_scene_dir, sep, dir)
                    thermal_image = "%s%sTherm_%s.tif" % (landsat_scene_dir, sep, dir)
                    angles_image = "%s%sAngl_%s.tif" % (landsat_scene_dir, sep, dir)
                    saturation_image = "%s%sSatu_%s.tif" % (landsat_scene_dir, sep, dir)
                    toa_image = "%s%sTOA_%s.tif" % (landsat_scene_dir, sep, dir)  # Top of atmosphere image
                    clouds_image = "%s%sClou_%s.tif" % (landsat_scene_dir, sep, dir)

                    # PYTHON COMMAND LINE CALCULATIONS
                    gdal_merge_refl = "python gdal_merge.py -separate -of HFA -co COMPRESSED=YES -o %s %s" % \
                                      (reflectance_image, " ".join(reflective_bands))
                    gdal_merge_therm = "python gdal_merge.py -separate -of HFA -co COMPRESSED=YES -o %s %s" % \
                                       (thermal_image, " ".join(thermal_bands))
                    # The next step is to create an image of the relevant angles, per-pixel, for use in subsequent steps. For
                    #   Landsat, this can be done using:
                    angles_per_pixel = 'python "%s" -m %s -t %s -o %s' % \
                                       (fpath_angles, mtl_file, reflectance_image, angles_image)

                    # Then mask and Top of Atmosphere reflectance must be calculated and finally the cloud mask itself:
                    # Saturation Image
                    saturation_mask = 'python "%s" -i %s -m %s -o %s' % \
                                      (fpath_satMask, reflectance_image, mtl_file, saturation_image)

                    toa = 'python "%s" -i %s -m %s -z %s -o %s' % \
                          (fpath_toa, reflectance_image, mtl_file, angles_image, toa_image)

                    # Clouds Stack image
                    clouds = 'python "%s" -t %s -a %s -m %s -z %s -s %s -o %s' % \
                             (fpath_stack, thermal_image, toa_image, mtl_file, angles_image, saturation_image,
                              clouds_image)

                    buildIfNotExist("reflectance stack", dir, reflectance_image, gdal_merge_refl)
                    buildIfNotExist("thermal stack", dir, thermal_image, gdal_merge_therm)
                    buildIfNotExist("angles", dir, angles_image, angles_per_pixel)
                    buildIfNotExist("saturation mask", dir, saturation_image, saturation_mask)
                    buildIfNotExist("toa", dir, toa_image, toa)
                    buildIfNotExist("clouds", dir, clouds_image, clouds)

                    print("Reading in cloud mask image...")
                    with rio.open(clouds_image) as cloud_eval:
                        """
                        cloud_image maps to
                        0 = Null
                        1 = Valid
                        2 = Cloud
                        3 = Cloud Shadow
                        4 = Snow
                        5 = Water
                        """

                        clouds = cloud_eval.read(1).astype(rio.uint8)

                    mask_clouds = clouds == 2
                    mask_cloud_shadows = clouds == 3

                    with rio.open(toa_image) as toa:
                        # toa nodata is 32767. written as 16 bit unsigned with all values multiplied by 10000
                        toa_band = toa.read()
                        kwargs = toa.profile

                        print("Setting cloud and cloud shadow values to 0")
                        # print(i)
                        toa_band = np.where(mask_clouds, 32767, toa_band)
                        toa_band = np.where(mask_cloud_shadows, 32767, toa_band)

                        kwargs.update(
                            nodata=32767,
                            compress='lzw',
                            dtype=rio.int16
                        )

                        print("Writing output file", ofile_path)
                        with rio.open(ofile_path, 'w', **kwargs) as dst:
                            dst.write(toa_band.astype(rio.int16))


def getLandsatScenes(landsat_dir, acquistionDate_raster, aoi_df, wrs_df, year, cloud_limit):
    with rio.open(acquistionDate_raster) as raster:
        nd_val = raster.nodata
        acqui_dates = raster.read(1).astype(float)
        doys = np.unique(acqui_dates).tolist()
        doys.remove(nd_val)  # remove nodata values
        try:
            doys.remove(0.0)
        except:
            pass

        beginning = int(min(doys) - 30)     # minimum acquisition date minus a month for more range
        end = int(max(doys) + 30)           # maximum acquisition date plus a month for more range

        print("START:", beginning)
        print("END: ", end)
        start_date = doyToDate(beginning, year)
        end_date = doyToDate(end, year)

        print("SEARCHING ALL LANDSAT SCENES BETWEEN %s AND %s" % (start_date, end_date))


    # GET INTERSECTION OF AOI AND LANDSAT PATHS, ROWS
    wrs_intersection = wrs_df[wrs_df.intersects(aoi_df.geometry[0])]

    paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values

    # the fmask algroithm used to remove cloud and cloud shadows can't operate on edges of
    #  landsate scenes. Clip by path buffer
    createLandsatPaths_and_Buffer(wrs_intersection, landsat_dir)

    print("Number of Paths: ", len(paths), ", Number of Rows: ", len(rows))

    scene_list_csv = os.path.join(landsat_dir, "scene_list_all.csv")
    if os.path.exists(scene_list_csv):
        print("FOUND FILE")
        s3_scenes = pd.read_csv(scene_list_csv)
    else:
        print("Scene list not found. Reading in from server")
        s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/scene_list.gz', compression='gzip')

    pd.options.display.max_colwidth = 255       # ensure that long data frame columns (e.g. urls aren't suppressed)

    #print("HEAD\n", s3_scenes.head(1))

    # Create date column and convert to date
    s3_scenes["date"] = s3_scenes.acquisitionDate.str[:10]
    s3_scenes["date"] = pd.to_datetime(s3_scenes['date'], format='%Y-%m-%d')

    # Filter data frame by dates
    filtered_scenes = s3_scenes[(s3_scenes.processingLevel.str.contains("L1T")) &
                                (s3_scenes['date'] >= start_date) &
                                (s3_scenes['date'] <= end_date) &
                                (s3_scenes.cloudCover <= cloud_limit)]

    frames = []

    for path, row in zip(paths, rows):
        #print('Path:',path, 'Row:', row)
        # Filter the Landsat Amazon S3 table for images matching path, row, cloud cover and processing state.
        path_row_scenes = filtered_scenes[(filtered_scenes.path == path) & (filtered_scenes.row == row)]
        #print(' Found %d images\n' % (len(path_row_scenes)))


        frames.append(path_row_scenes)
        #print("Number of images: ", path_row_scenes.shape[0])

    bulk_frame = pd.concat(frames)
    total_scenes = len(bulk_frame)
    print("Total number of Landsat Scenes for all dates in area: ", total_scenes)
    return bulk_frame


def createLandsatPaths_and_Buffer(geodataframe, landsat_dir):
    """ Takes the wrs2 paths and row features passed as dataframe and dissolves them by path.
    Dissolved paths are then buffered by -0.05 decimal degrees. Finally, each feature is
    saved to its own shapfile in the same directory as the wrs2 shapefile"""
    outdir = os.path.join(landsat_dir, "dissolvedPaths")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    wkt_32612 = '+proj=utm +zone=12 +ellps=WGS84 +datum=WGS84 +units=m +no_defs '
    geodataframe_32612 = geodataframe.to_crs(wkt_32612)

    geodataframe_dissolve = geodataframe_32612.dissolve("PATH")
    geodataframe_dissolve['geometry'] = geodataframe_dissolve.geometry.buffer(-4000)

    for i in range(0, len(geodataframe_dissolve)):  # first_feature = wrs_dissolve[0:1]
        feat = geodataframe_dissolve[i:i + 1]
        outname = "path" + str(feat.index[0]) + "_buffered.shp"
        outfile = os.path.join(outdir, outname)

        feat.to_file(outfile)


def getDissolvedPathShapefile(landsat_path, landsat_dir):
    paths_dir = os.path.join(landsat_dir, "dissolvedPaths")
    for f in os.listdir(paths_dir):
        if landsat_path in f and f.endswith(".shp"):
            fp = os.path.join(paths_dir, f)
            return fp


def clipDOYFiles(in_dir, out_dir, acqui_raster, landsat_dir):
    # print(os.listdir(dir))

    temp_folder = "temp_resized"
    temp_dir = os.path.join(out_dir, temp_folder)
    utilities.useDirectory(temp_dir)

    tifs = [f for f in glob.iglob(in_dir + "/*.tif")]
    for file in tifs:
        count = 0

        fname = os.path.splitext(file)[0]

        temp_ras = os.path.join(temp_dir, os.path.basename(fname) + "_clipped_resized.tif")
        trans_ras = os.path.join(out_dir, os.path.basename(fname) + "_clipped.tif")

        if not os.path.exists(trans_ras):
            count = 0
            count += 1
            landsat_path = int(fname[-2:])
            vector = getDissolvedPathShapefile(str(landsat_path), landsat_dir)

            # CLIP MERGE RASTER PATH TO REMOVE EDGES WHICH HAVE UNMAKSED CLOUDS
            print("%d - Starting clip of %s to %s" % (count, file, temp_ras))
            gdal_clip = "gdalwarp -overwrite -srcnodata 32767 -dstnodata 32767 -tr 30 30 -crop_to_cutline" \
                        " -cutline %s %s %s" % (vector, file, temp_ras)
            os.system(gdal_clip)

            data = gdal.Open(acqui_raster, GA_ReadOnly)
            projwin_srs = data.GetProjection()

            geoTransform = data.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * data.RasterXSize
            miny = maxy + geoTransform[5] * data.RasterYSize

            # NEED TO RESIZE RASTER TO FIT WHOLE EXTENT
            print("\tStarting Translate of ", temp_ras)
            projwin = str(minx) + " " + str(maxy) + " " + str(maxx) + " " + str(miny)
            gdaltranslate = 'gdal_translate -projwin ' + projwin + " -projwin_srs " + projwin_srs + " -of GTiff " + temp_ras + " " + trans_ras
            os.system(gdaltranslate)

            os.remove(temp_ras)


def rasterizeVector(invec, outras):
    if not os.path.exists(outras):
        if os.path.exists(invec):
            print("Rasterizing naip acquisition vector...")
            rasterize_acqui_vector = 'gdal_rasterize -tap -a_nodata %d -ot Int16 -tr 30 30 -a "DOY" "%s" "%s"' % \
                                     (32766, invec, outras)
            os.system(rasterize_acqui_vector)
        else:
            print("Unable to locate given vector file %s. Unable to complete rasterization" % invec)
            raise Exception
    else:
        print("Raster %s already exists. Skipping rasterization from %s." % (outras, invec))


def getLandsatTOA(landsat_dir, acqui_raster, aoi_file, wrs_file, cloud_limit, year, target_epsg="32612"):
    """ Central function initiates download of landsat tiles
    which coincide with the NAIP acquisition days from Amazon
     S3 and then runs and fmask TOA and Cloud removal algorythm"""

    LT1_dir = os.path.join(landsat_dir, "LT1")
    utilities.useDirectory(LT1_dir)
    toa_dir = os.path.join(landsat_dir, "TOA_Corrected")
    utilities.useDirectory(toa_dir)

    aoi_df = gpd.read_file(aoi_file)
    wrs_df = gpd.GeoDataFrame.from_file(wrs_file)

    scenes = getLandsatScenes(landsat_dir, acqui_raster, aoi_df, wrs_df, year, cloud_limit)
    downloadList(scenes, LT1_dir)

    # RUN TOA CALCULATION ON NOW DOWNLOADED LANDSAT FILES
    createCloudMaskedTOA(LT1_dir, toa_dir)

    findAndReproject(toa_dir, target_epsg)



if __name__ == "__main__":
    # IMPORTANT input values for landsat data
    year = 2015
    cloud_limit = 40

    data_dir = os.path.abspath(r"M:/Data")

    landsat_dir = os.path.join(data_dir, "Landsat8")
    utilities.useDirectory(landsat_dir)

    # Specify the Area of interest file and landsat path/row file (wrs2). An intersect will find all necessary path/rows
    aoi_file_loc = os.path.join(data_dir, "NAIP_AZ_Boundary.gpkg")

    landsat_wrs2_loc = os.path.join(landsat_dir, "wrs2_descending.shp")
    wrs = gpd.GeoDataFrame.from_file(landsat_wrs2_loc)

    acqui_vector = os.path.abspath(r"Q:\Arid Riparian Project\Data\NAIP2015_AcquisitionDates_Dissolve.gpkg")
    acqui_raster = os.path.abspath(r"M:\Data\NAIP2015_AcquisitionDates_Dissolve.tif")

    rasterizeVector(acqui_vector, acqui_raster)

    use_EPSG = "32612"

    getLandsatTOA(landsat_dir, acqui_raster, aoi_file_loc, landsat_wrs2_loc, cloud_limit, year=year, target_epsg=use_EPSG)


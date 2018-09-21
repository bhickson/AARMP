from __future__ import print_function

# %matplotlib inline

from joblib import Parallel, delayed

import os
import shutil
import rasterio as rio
import numpy as np
from collections import OrderedDict

from datetime import datetime
from skimage.segmentation import felzenszwalb

from skimage import measure
from skimage.measure import label

import geopandas as gpd

import gdal
import logging as logger
import fiona

from Utilities import getFullNAIPPath
import Utilities as utils

from irods.session import iRODSSession
from irods.models import Collection, DataObject

from shapely.geometry import Polygon, MultiPolygon

import time


def segmentImage(loc_NAIPFile, seg_img_dir, overwrite=False, return_data=True):
    # print("\tStarting segmentation of NAIP File: %s" % loc_NAIPFile)

    ofile_name = os.path.basename(loc_NAIPFile)[:-4] + "_Segmented.tif"

    o_file = os.path.join(seg_img_dir, ofile_name)

    if not os.path.exists(o_file) or overwrite:
        seg_start = datetime.now()
        start = datetime.now()
        # print("Segmented file doesn't exist or overwrite set. Creating at %s" % o_file)

        with rio.open(loc_NAIPFile) as inras:
            prof = inras.profile

            ras_array = inras.read()

            ras_array = np.moveaxis(ras_array, 0, -1)  # scikit-image wants array in (height, width, bands). Rasterio returns (bands, height, width)
            bands_array_seg = segmentArray(ras_array)

        prof.update(
            count=1,
            dtype=np.int32
        )

        with rio.open(o_file, 'w', **prof) as oras:
            oras.write_band(1, bands_array_seg.astype(np.int32))

        seg_end = datetime.now()
        print("\tSegmentation took {}".format(seg_end - seg_start))

    # pushToDE(o_file, irods_files, irods_sess)

    if return_data:
        with rio.open(o_file) as ras:
            seg_array = ras.read(1).astype(np.int32)

        return seg_array


def segmentArray(image_array, seg_type="felzenswalb"):
    print("Beginning image segmentation on array")
    seg_start = datetime.now()
    if seg_type == "felzenswalb":
        segments_fz = felzenszwalb(image_array, scale=25, sigma=0.5, min_size=3, multichannel=True)
    else:
        print("Unknown segmentation algorithm. Exiting...")
        raise ValueError

    seg_end = datetime.now()
    print(
        "Felzenszwalb number of segments: {}. \n\t{} elapsed.".format(len(np.unique(segments_fz)), seg_end - seg_start))

    return segments_fz


def calcSegmentMean(labeled_array, regs, in_band):
    mean_start = datetime.now()
    # label_im = label(segments, connectivity=1) + 1
    # regions = measure.regionprops(labeled_array.astype(np.int64), intensity_image=in_band)

    print("\tBeginning mean calculation on segments...")
    #mean_array = np.copy(labeled_array)

    values_map = {}
    for i,r in enumerate(regs):
        values_map[r.label] = r.mean_intensity

    mean_array = vec_translate(labeled_array, values_map)
    #print("Full Vectorization: ", datetime.now() - vec_start)

    
    """
    for r in regs:
        segment = r.label
        bbox = r.bbox
        min_row, max_row, min_col, max_col = bbox

        sub_array = mean_array[min_row:min_col, max_row:max_col]
        # print(sub_array)

        sub_array[sub_array == segment] = r.mean_intensity
        # print(sub_array)

        mean_array[min_row:min_col, max_row:max_col] = sub_array
    """
    print("\t...Mean calculation complete.\n\t{} elapsed.".format(datetime.now() - mean_start))

    return mean_array


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def calculateGeometry(seg_array, regs):
    start = datetime.now()
    # regions = measure.regionprops(labeled_array.astype(np.int32))  #, intensity_image=empty_ar)
    # print("got regions")

    labeled_array = label(seg_array, connectivity=1).astype(np.float32) + 1

    #out_array = np.copy(labeled_array)

    value_map_area = {}
    value_map_perim = {}
    value_map_extent = {}

    for i,r in enumerate(regs):
        value_map_area[r.label] = r.area
        value_map_perim[r.label] = r.perimeter
        value_map_extent[r.label] = r.extent

    area_array = vec_translate(labeled_array, value_map_area)
    perim_array = vec_translate(labeled_array, value_map_perim)
    perc_area_array = vec_translate(labeled_array, value_map_extent)

    return {"area": area_array, "perim": perim_array, "perc_area": perc_area_array}
    """
    for r in regs:
        segment = r.label
        bbox = r.bbox
        min_row, max_row, min_col, max_col = bbox
        sub_array = labeled_array[min_row:min_col, max_row:max_col]

        if attrib.lower() == "area":
            value = r.area
        elif attrib.lower() == "perim":
            value = r.perimeter
        elif attrib.lower() == "perc_area":
            value = r.extent * 100

        sub_array[sub_array == segment] = value

        out_array[min_row:min_col, max_row:max_col] = sub_array

    return out_array
    """


def vegIndexCalc(naip_array_list, indicies):
    print("\tBeginning VegIndexCalcs")
    bandRed = naip_array_list[0].astype(float)
    bandGreen = naip_array_list[1].astype(float)
    bandBlue = naip_array_list[2].astype(float)
    bandNIR = naip_array_list[3].astype(float)

    # IMPORTANT: Because of the soil line value in the SAVI indicies,
    #  all band value must be normalized between 0 and 1.
    bandRed /= 255.0
    bandNIR /= 255.0
    bandBlue /= 255.0

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    vi_calcs = OrderedDict()
    for veg_index in indicies:

        if veg_index == "NDVI":
            vi = (bandNIR - bandRed) / (bandNIR + bandRed)

        elif veg_index == "SAVI":
            l = 0.5
            vi = ((bandNIR - bandRed) / (bandNIR + bandRed + l)) * (1 + l)

        elif veg_index == "MSAVI2":
            vi = ((2 * bandNIR + 1) - np.sqrt(np.square(2 * bandNIR + 1) - (8 * (bandNIR - bandRed)))) / 2

        elif veg_index == "EVI2":
            g = 2.5  # gain factor
            l = 1.0  # soil adjustment factor
            c_one = 6.0  # coefficient
            c_two = 7.5  # coefficient
            # vi = 2.5 * ((bandNIR - bandRed) / (bandNIR + (2.4 * bandRed) + 1))
            vi = g * ((bandNIR - bandRed) / ((bandNIR + c_one) * (bandRed - c_two) * (bandBlue + l)))

        elif veg_index == "OSAVI":
            vi = (bandNIR - bandRed) / (bandNIR + bandRed + 0.16)

        vi_calcs[veg_index] = vi * 1000  # multiply by 1000 so we can convert to int16 later without loosing precision

        del vi

    return vi_calcs


def getSubSetSlope(naip_path, slope_file, odir, overwrite=False):
    ssl_start = datetime.now()
    ofile = "SlopeDeg_" + os.path.basename(naip_path)

    slope_file = os.path.abspath(slope_file)

    slope_opath = os.path.join(odir, ofile)

    print(slope_opath)
    if not os.path.exists(slope_opath) or overwrite:
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

        # build slope qquad from naip extent

        resampletype = "bilinear"

        gdal_warp = "gdalwarp -overwrite -tap -r %s -t_srs %s -tr %s %s -te_srs %s -te %s %s %s %s %s %s" % (
            resampletype, proj, resx, resy, proj, str(minx), str(miny), str(maxx), str(maxy), slope_file,
            slope_opath)
        # logger.info("\tExecuting gdal_warp operation on %s for footprint of naip file %s" % (slope_file, naip_path))
        print("Executing gdal_warp operation on %s for footprint of naip file %s" % (slope_file, naip_path))

        os.system(gdal_warp)
        # logger.info("\tFinished qquad for %s landsat in %s" % (slope_file, str(datetime.now() - ssl_start)))

    with rio.open(slope_opath) as s_ras:
        s_ras_array = s_ras.read()

    # File cleanup to save space
    os.remove(slope_opath)

    # multiply by 100 conserve precision when reducing to int16 (e.g. 40.0215 degrees -> 4021)
    s_ras_array *= 100

    print("Bringing in slope array from %s ..." % slope_opath)
    return s_ras_array.astype(np.int16)


def getSubSetLandsat(naip_path, landsat_file, opath, overwrite=False, return_data=True):
    ssl_start = datetime.now()
    ofile = "Landsat8_" + os.path.basename(naip_path)

    landsat_opath = os.path.join(opath, ofile)
    try:
        if not os.path.exists(landsat_opath) or overwrite:
            if ofile in irods_files.keys() and return_data == True:
                # download from de

                irods_path = irods_files[ofile]
                # downloadFromDE(irods_path, landsat_opath)
                get_command = "iget -K " + irods_path
                print("Downloading from iRods DE...")
                os.system(get_command)

                # shutil.move(ofile, landsat_opath)
                with rio.open(ofile) as lras:
                    lras_array = lras.read()

                os.remove(ofile)

                if "ndsi" in opath.lower() or "ndwi" in opath.lower():
                    lras_array = lras_array * 1000

                print("Bringing in landsat array from %s ..." % opath)
                return lras_array.astype(np.int16)

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
            logger.info(
                "\tExecuting gdal_warp operation on %s for footprint of naip file %s" % (landsat_file, naip_path))
            print("Executing gdal_warp operation on %s for footprint of naip file %s" % (landsat_file, naip_path))

            os.system(gdal_warp)
            logger.info("\tFinished qquad for %s landsat in %s" % (landsat_file, str(datetime.now() - ssl_start)))

        if return_data == True:
            print("Bringing in landsat array from %s ..." % opath)
            with rio.open(landsat_opath) as lras:
                lras_array = lras.read()

            # File cleanup to save space
            os.remove(landsat_opath)

            if "ndsi" in opath.lower() or "ndwi" in opath.lower():
                lras_array = lras_array * 1000

            return lras_array.astype(np.int16)
    except:
        lras_array = getSubSetLandsat(naip_path, landsat_file, opath, overwrite=True, return_data=False)
        return lras_array.astype(np.int16)


def compoundArrays(array_stack, tiff_tags, arrays_dict):
    for name, array in arrays_dict.items():
        tiff_tags[len(tiff_tags) + 1] = name

        array_stack.append(array.astype(np.int16))

    return array_stack, tiff_tags


def generateStack(loc_NAIPFile, base_dir=r"../Data", veg_indicies=["NDVI", "SAVI", "OSAVI", "MSAVI2", "EVI2"],
                  overwrite=False):  # , training_stack_dir, naip_band_order, veg_indicies):
    print("Starting on NAIP File: %s" % loc_NAIPFile)

    # Identify and output directory - create if necessary
    base_datadir = os.path.abspath(base_dir)
    training_stack_dir = utils.useDirectory(os.path.join(base_datadir, "TrainingImageStack"))
    segmentedImagesDir = utils.useDirectory(os.path.join(base_datadir, "SegmentedNAIPImages"))
    base_landsatdir = utils.useDirectory(os.path.join(base_datadir, "Landsat8"))
    ndsi_qquad_dir = utils.useDirectory(os.path.join(base_datadir, "NDSI"))
    ndwi_qquad_dir = utils.useDirectory(os.path.join(base_datadir, "NDWI"))
    landsat_qquad_dir = utils.useDirectory(os.path.join(base_landsatdir, "byNAIPDOY_QQuads"))
    slope_qquad_dir = utils.useDirectory(os.path.join(base_datadir, "Slope"))

    landsat_file = os.path.os.path.join(base_landsatdir, "Landsat1to8_TOA_NAIPAcquiDate_merge_rectified.tif")
    # LOCATION OF THE NDSI FILE
    ndsi_file = os.path.join(ndsi_qquad_dir, "LandsatOLI_NDSI_30m.tif")
    # LOCATION OF THE NDWI FILE
    ndwi_file = os.path.join(ndwi_qquad_dir, "LandsatOLI_NDWI_30m.tif")

    slope_file = os.path.join(slope_qquad_dir, "Slope-Degrees_AZ.tif")

    if not os.path.exists(training_stack_dir):
        os.mkdir(training_stack_dir)

    # Specify the order of the NAIP bands - should never change
    naip_band_order = {1: "RED", 2: "GREEN", 3: "BLUE", 4: "NIR"}

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ofile_name = os.path.basename(loc_NAIPFile)[:-4] + "_TrainingStack.tif"
    o_file = os.path.join(training_stack_dir, ofile_name)

    irods_data_path = "/iplant/home/bhickson/2015/Data"
    irods_sess, irods_files = getFilesonDE(irods_data_path)

    # IF FILE ON DE, DOWNLOAD
    # BEN YOU PUT THIS HERE TO ACCOUNT FOR SLOPE OVERWRITE. DE FILE AND LOCAL FILE MAY HAVE 22 BANDS
    if not os.path.exists(o_file):
        if ofile_name in irods_files.keys():
            print("Found {} in Cyverse DE. Downloading...".format(ofile_name))
            # download from de
            irods_path = irods_files[ofile_name]
            # downloadFromDE(irods_path, landsat_opath)
            get_command = "iget -K " + irods_path

            os.system(get_command)

            shutil.move(ofile_name, training_stack_dir)

    if not os.path.exists(o_file) or overwrite:
        start = datetime.now()
        print("Training stack doesn't exist. Creating at %s" % o_file)

        bands_array_seg = segmentImage(loc_NAIPFile, segmentedImagesDir)

        with rio.open(loc_NAIPFile) as ras:
            prof = ras.profile

            # initialize = rio.open(o_file, 'w', **prof).close()  # prevent parallel process for working on same file

            n_bands = prof["count"]

            output_array_stack = []

            tags = dict(naip_band_order)
            # ras_array = ras.read()
            # ras_array = np.moveaxis(ras_array, 0, -1)  # scikit-image wants array in (height, width, bands). Rasterio returns (bands, height, width)

            # bands_array_seg = segmentArray(ras_array)

            # Identify segments as individual labels
            label_im = label(bands_array_seg, connectivity=1).astype(np.float32) + 1

            # Iterate through R,G,B,NIR bands and use the segmented image to calculate mean in zones/labels
            for b in range(1, n_bands + 1):
                band_array = ras.read(b)

                regions = measure.regionprops(label_im.astype(np.int32), intensity_image=band_array)

                seg_mean_band = calcSegmentMean(in_band=band_array, regs=regions, labeled_array=label_im)

                output_array_stack.append(seg_mean_band.astype(np.int16))

            # CREATED BANDS FOR SEGMENT SIZE CHARACERISTICS (Area/Perimeter of segment, and % area of bounding box)
            """for att in ["area", "perim", "perc_area"]:
                print("\tStarting %s" % att)
                oa = calculateGeometry(bands_array_seg, regions, att)
                tags[len(tags) + 1] = att
                output_array_stack.append(oa.astype(np.int16))"""
            geometric_attribs_arrays = calculateGeometry(bands_array_seg, regions)
            output_array_stack.append(geometric_attribs_arrays["area"])
            output_array_stack.append(geometric_attribs_arrays["perim"])
            output_array_stack.append(geometric_attribs_arrays["perc_area"])

            del geometric_attribs_arrays
            del bands_array_seg
            del label_im
            # geometric_arrays = calculateGeometry(labeled_array=label_im, regs=regions)


            # output_array_stack, tags = compoundArrays(output_array_stack, tags, geometric_arrays)

            # CREATE VEG INDEX ARRAYS
            veg_arrays_dict = vegIndexCalc(output_array_stack, veg_indicies)
            output_array_stack, tags = compoundArrays(output_array_stack, tags, veg_arrays_dict)

            # CREATE LANDSAT INDEX ARRAYS AND APPEND TO MAIN ARRAY WITH TAGS
            # GET ALL LANDSAT BAND ARRAYS AND APPEND TO MAIN ARRAY
            landsat_qquad_array = getSubSetLandsat(loc_NAIPFile, landsat_file, landsat_qquad_dir, overwrite=False)
            for i in range(len(landsat_qquad_array)):
                output_array_stack.append(landsat_qquad_array[i].astype(np.int16))
                tags[len(tags) + 1] = "L8_" + str(i + 1)

            # APPEND NDSI ARRAY TO MAIN ARRAY
            landsat_ndsi_array = getSubSetLandsat(loc_NAIPFile, ndsi_file, ndsi_qquad_dir, overwrite=False)
            output_array_stack.append(landsat_ndsi_array[
                                          0])  # full read has shape (1, 7500, 6900). Must be 7500,6900 for equal shape of other arrays
            tags[len(tags) + 1] = "L8_NDSI"

            # APPEND NDWI ARRAY TO MAIN ARRAY
            landsat_ndwi_array = getSubSetLandsat(loc_NAIPFile, ndwi_file, ndwi_qquad_dir, overwrite=False)
            output_array_stack.append(landsat_ndwi_array[0])
            tags[len(tags) + 1] = "L8_NDWI"

            slope_array = getSubSetSlope(loc_NAIPFile, slope_file, slope_qquad_dir)
            output_array_stack.append(slope_array[0])
            tags[len(tags) + 1] = "Slope_Degrees"
            del slope_array

            out_array_stack_np = np.stack(output_array_stack, axis=0)
            # print(out_array_stack_np.shape)
            # print(out_array_stack_np.dtype)

        prof.update(
            dtype=rio.int16,
            count=len(out_array_stack_np)
        )

        with rio.open(o_file, 'w', **prof) as outras:
            for n, tag in tags.items():
                outras.update_tags(n, NAME=tag)
            outras.write(out_array_stack_np.astype(rio.int16))

        end = datetime.now()
        print("\tFINISHED -\t%s Elapsed - %s" % (str(end - start), o_file))

    else:
        # if file already exists, open it, get slope and append
        print("Adding on slope array to existing training stack: {}".format(o_file))
        with rio.open(o_file) as orig_stack:
            kwargs = orig_stack.profile
            if not orig_stack.count == 23:
                twentythree = False
                kwargs.update(count=23)
                orig_stack_array = orig_stack.read()
                tags = {}
                for i in range(1, orig_stack.count + 1):
                    tags[i] = orig_stack.tags(i)["NAME"]
            else:
                twentythree = True

        if not twentythree:
            with rio.open(o_file, 'w', **kwargs) as new_stack:
                slope_array = getSubSetSlope(loc_NAIPFile, slope_file, slope_qquad_dir)
                out_array = np.concatenate((orig_stack_array, slope_array), axis=0)
                for n, tag in tags.items():
                    new_stack.update_tags(n, NAME=tag)
                new_stack.update_tags(23, NAME="Slope_Degrees")
                new_stack.write(out_array.astype(np.int16))

    """except ValueError:
        print("\n----------- SHAPE ERROR-----------\n")
        print("\t", loc_NAIPFile)
        print("\tSEGMENT SHAPE:", bands_array_seg.shape)
        print("\tBAND SHAPE:", band_array.shape)
        print("\tLABEL SHAPE: {}\n".format(label_im.shape))
        return
    """
    return o_file


def getFilesonDE(base_path):
    pw_file = "./pw_file.txt"
    try:
        with open(pw_file) as pf:
            pw = pf.readlines(0)[0].strip()

        session = iRODSSession(host='data.cyverse.org', zone="iplant", port=1247, user='bhickson', password=pw)

        data_col = session.collections.get(base_path)

    except:
        print("Unable to make connection to discover env. Continuing...")
        return None, {}

    ifiles = {}

    def getFilesandDirs(dir):
        # print(dir.name)
        files_list = dir.data_objects
        dirs_list = dir.subcollections
        for file in files_list:
            file_name = file.name
            ifiles[file.name] = file.path
        for sub_dir in dirs_list:
            # print(sub_dir.name)
            getFilesandDirs(sub_dir)

    getFilesandDirs(data_col)

    return session, ifiles


def pushToDE(file_path, irods_files, session, base_datadir="../Data",
             irods_data_path="/iplant/home/bhickson/2015/Data"):
    # push comleted file to irods
    fname = os.path.basename(file_path)
    if fname not in irods_files.keys():
        relative_dir = file_path.split(os.path.abspath(base_datadir))[1]
        irods_odir = irods_data_path + relative_dir
        print("-------- Pushing data to DE....  -   %s" % irods_odir)
        session.data_objects.put(file_path, irods_odir)


if __name__ == '__main__':
    print(datetime.now())

    logger.basicConfig(level=logger.INFO)

    veg_indicies = ["NDVI", "SAVI", "OSAVI", "MSAVI2", "EVI2"]

    base_datadir = os.path.abspath(r"../Data")
    vectorinputs_dir = os.path.join(base_datadir, "initial_model_inputs")
    base_landsatdir = os.path.join(base_datadir, "Landsat8")
    naip_dir = os.path.join(base_datadir, "NAIP")
    ndsi_qquad_dir = os.path.join(base_datadir, "NDSI")
    ndwi_qquad_dir = os.path.join(base_datadir, "NDWI")
    landsat_qquad_dir = utils.useDirectory(os.path.join(base_landsatdir, "byNAIPDOY_QQuads"))

    landsat_file = os.path.os.path.join(base_landsatdir, "Landsat1to8_TOA_NAIPAcquiDate_merge_rectified.tif")
    # LOCATION OF THE NDSI FILE
    ndsi_file = os.path.join(ndsi_qquad_dir, "LandsatOLI_NDSI_30m.tif")
    # LOCATION OF THE NDWI FILE
    ndwi_file = os.path.join(ndwi_qquad_dir, "LandsatOLI_NDWI_30m.tif")

    training_stack_dir = os.path.join(base_datadir, "TrainingImageStack")
    utils.useDirectory(training_stack_dir)
    segmentedImagesDir = os.path.join(base_datadir, "SegmentedNAIPImages")
    utils.useDirectory(segmentedImagesDir)

    print("Creating irods files dict...")
    global irods_files

    irods_data_path = "/iplant/home/bhickson/2015/Data"
    irods_sess, irods_files = getFilesonDE(irods_data_path)
    exit()
    #irods_fils = {}
    # for f, p in irods_files.items():
    #    print("FNAME:", f, "PATH: ", p)
    """
    print("\n--------------- Starting with Subset QQuads ---------------\n")
    single_comp_subset = []
    with open(os.path.join(base_datadir,"NaipDone4.txt")) as txt:
        for l in txt:
            #print(l)
            fullname = ""#getFullNAIPPath(l[:-1], naip_dir)
            single_comp_subset.append(fullname)

    def processLS(f, rd=False):
        getSubSetLandsat(f, ndsi_file, ndsi_qquad_dir, return_data=rd)
        getSubSetLandsat(f, ndwi_file, ndwi_qquad_dir, return_data=rd)
        getSubSetLandsat(f, landsat_file, landsat_qquad_dir, return_data=rd)

    ef = []
    for root,dirs,files in os.walk(naip_dir):
        for file in files:
            if file.endswith(".tif"):
                f = os.path.join(root,file)
                ef.append(f)

    Parallel(n_jobs=-1, max_nbytes=None, verbose=40, backend='loky', temp_folder=segmentedImagesDir) \
        (delayed(processLS)(naip_file, rd=False) for naip_file in ef)

    #for f in glob.glob(landsat_qquad_dir + "/" + ".tif"):
    #    fpath = os.path.join(landsat_qquad_dir, f)
    #    pushToDE(fpath, irods_files, irods_sess)


    Parallel(n_jobs=8, max_nbytes=None, verbose=40, backend='loky', temp_folder=segmentedImagesDir) \
        (delayed(generateStack)(naip_file) for naip_file in ef)
    for root,dirs,files in os.walk(naip_dir):
        for file in files:
            fpath = os.path.join(root,files)
            pushToDE(fpath, irods_files, irods_sess)


    print("\n--------------- Finished with Subset QQuads ---------------\n")
    """
    """
    print("\n--------------- Starting with training qquads ---------------\n")
    print("Reading in class_points_file...")
    loc_class_points = os.path.join(vectorinputs_dir, "classificationPoints.shp")
    training_data_df = gpd.read_file(loc_class_points, crs={'init': 'epsg:26912'})

    count = 0
    training_naip_files = []
    for loc_NAIPFile, group in training_data_df.groupby("NAIP_FILE"):
        # print(loc_NAIPFile)
        count += 1
        # print(count, " - ", os.path.basename(loc_NAIPFile))
        training_naip_files.append(loc_NAIPFile)
    del training_data_df

    #for naip_file in training_naip_files:
    #    generateStack(naip_file)

    # Parallel(n_jobs=4, max_nbytes=None, verbose=30, backend='loky', temp_folder=segmentedImagesDir)\
    #    (delayed(segmentImage)(naip_file, segmentedImagesDir, return_data=False) for naip_file in training_naip_files)
    Parallel(n_jobs=4, max_nbytes=None, verbose=40, backend='loky', temp_folder=segmentedImagesDir) \
        (delayed(generateStack)(naip_file) for naip_file in training_naip_files)

    # Parallel(n_jobs=8)(delayed(generateStack) (naip_file) for naip_file in training_naip_files)

    print("\n--------------- Finished with training qquads ---------------\n")
    """
    print("\n--------------- Starting on aoi qquads ---------------\n")
    base_datadir = os.path.abspath(r"../Data")
    training_data_dir = os.path.join(base_datadir, "initial_model_inputs")
    loc_usgs_qquads = os.path.join(training_data_dir, "USGS_QQuads_AZ.shp")
    footprints = gpd.read_file(loc_usgs_qquads)
    # aoi = gpd.GeoDataFrame.from_file(r"Q:\Arid Riparian Project\AridRiparianProject\AridRiparianProject.gdb", layer='TargetEcoregions')

    # aoi.crs = fiona.crs.from_epsg(2163)
    # print(aoi.crs)
    aoi = gpd.read_file(os.path.join(vectorinputs_dir, "Maricopa_County.gpkg"))

    footprints = gpd.read_file(loc_usgs_qquads)
    aoi.to_crs(footprints.crs, inplace=True)
    
    irods_naip_sess, irods_naip_files = getFilesonDE(irods_data_path + "/NAIP")

    from glob import glob
    existing_ts = glob(training_stack_dir + "/*.tif")
    aoi_qquads = []
    for i, row in footprints.iterrows():
        for j, arow in aoi.iterrows():
            if row.geometry.intersects(arow.geometry):
                fpath = getFullNAIPPath(row.QUADID, r"../Data/NAIP", irods_naip_files)
                basename = os.path.basename(fpath)
                aoi_qquads.append(fpath)
                for f in existing_ts:
                    if basename.split(".")[0] in f:
                        aoi_qquads.remove(fpath)


    def processLS(f, rd=False):
        getSubSetLandsat(f, ndsi_file, ndsi_qquad_dir, return_data=rd)
        getSubSetLandsat(f, ndwi_file, ndwi_qquad_dir, return_data=rd)
        getSubSetLandsat(f, landsat_file, landsat_qquad_dir, return_data=rd)

    def tryGenerateStack(file):
        try:
            generateStack(file)
        except:
            print("Unable to segment {}".format(file))
            with open("failures.txt", "a+") as tf:
                file.write(tf + "\n")

    print("\nBeginning image segmentation for {} QQuads\n".format(len(aoi_qquads)))
    Parallel(n_jobs=4, max_nbytes=None, verbose=30, backend='loky', temp_folder=segmentedImagesDir) \
            (delayed(segmentImage)(naip_file, segmentedImagesDir) for naip_file in aoi_qquads)


    print("\nStarting stack generation for {} QQuads\n".format(len(aoi_qquads)))
    #for naip_file in aoi_qquads:
    #    generateStack(naip_file)
    Parallel(n_jobs=12, max_nbytes=None, verbose=40, backend='loky', temp_folder=segmentedImagesDir) \
        (delayed(tryGenerateStack)(naip_file) for naip_file in aoi_qquads)

    #for root, dirs, files in os.walk(naip_dir):
    #    for file in files:
    #        fpath = os.path.join(root, files)
    #        pushToDE(fpath, irods_files, irods_sess)

    # Parallel(n_jobs=4, max_nbytes=None, verbose=30, backend='loky', temp_folder=segmentedImagesDir) \
    #    (delayed(segmentImage)(naip_file, segmentedImagesDir) for naip_file in aoi_qquads)
    # Parallel(n_jobs=4, max_nbytes=None, verbose=30, backend='loky', temp_folder=segmentedImagesDir)\
    #    (delayed(generateStack)(naip_file) for naip_file in aoi_qquads)

    print("\n--------------- Finished with aoi qquads ---------------\n")

    exit()
    print("\n--------------- Starting with all qquads ---------------\n")
    all_naip_files = []
    for root, dirs, files in os.walk(r"../NAIP"):
        for file in files:
            if file.endswith(".tif"):
                fpath = os.path.join(root, file)
                all_naip_files.append(fpath)

    # Parallel(n_jobs=1)(delayed(segmentImage)(naip_file) for naip_file in all_naip_files)
    Parallel(n_jobs=4)(delayed(generateStack)(naip_file) for naip_file in all_naip_files)
    print("\n--------------- Finished with all qquads ---------------\n")

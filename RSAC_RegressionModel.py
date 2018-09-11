# Random Forests Regression Model to predict valley bottoms. Based loosely on the USFS RSAC model


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

    return [raster_names, raster_paths]


def createClassifiedFile(rasters, regmodel, loc_classified_file, overwrite=False):
    cl_start = datetime.now()

    if not os.path.exists(loc_classified_file) or overwrite:
        localLog.info("Creating classified file: %s..." % loc_classified_file)
        # GET RASTER INFO FROM INPUT
        # NEED TO GET BANDS DATA INTO SINGLE ARRAY FOR OUTPUT CLASSIFICATION
        # bands_data_rio = []
        bands_data = []
        for inras in rasters:
            localLog.info("Reading in raster file as array - %s" % inras)

            with rio.open(inras) as raster:
                kwargs = raster.profile
                b_array = raster.read(1).astype(rio.float32)

            bands_data.append(b_array)

        # CREATE NP DATASTACK FROM ALL RASTERS
        localLog.debug("Creating numpy array stack...")
        bands_data = np.dstack(bands_data)

        # print("BANDS_DATA.SHAPE: ", bands_data.shape)
        # CREATE VARIABLES OF ROWS, COLUMNS, AND NUMBER OF BANDS
        rows, cols, n_bands = bands_data.shape
        n_samples = rows * cols
        # print("N_Samples: ", n_samples)
        # print("n_bands: ", n_bands)

        # CREATE EMPTY ARRAY WITH SAME SIZE AS RASTER
        localLog.debug("Reshaping numpy array to raster shape...")
        flat_pixels = bands_data.reshape((n_samples, n_bands))

        localLog.debug("Predicting Valley Bottoms...")
        result = regmodel.predict(flat_pixels)

        # Reshape the result: split the labeled pixels into rows to create an image
        classification = result.reshape((rows, cols))

        # WRITE OUT THE CLASSIFIED ARRAY TO RASTER BASED ON PROPERTIES OF TRAINING RASTERS
        # write_geotiff(loc_classified_file, classification, geo_transform, proj, classes, COLORS)

        kwargs.update(
            dtype=rio.float32,
            nodata=1
        )
          
        with rio.open(loc_classified_file, 'w', **kwargs) as outras:
            outras.write_band(1, classification.astype(rio.float32))

        localLog.info("Classification created:\n\t %s in %s" % (loc_classified_file, str(datetime.now() - cl_start)))
    else:
        localLog.info("The file exists and no overwrite set. Skipping creating %s" % loc_classified_file)

    return loc_classified_file


def rasterSubDivide(preds_dir, overwrite=False):
    parent_dir = utils.getParentDir(preds_dir)
    outdir = os.path.join(parent_dir, "predictors_quads")
    utils.useDirectory(outdir)

    rasters = getRasterNamesList(preds_dir)[1]

    for raster in rasters:
        reference_f = gdal.Open(raster)
        geo_transform = reference_f.GetGeoTransform()
        resx = geo_transform[1]
        resy = geo_transform[5]
        proj = reference_f.GetProjectionRef()
        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + (resx * reference_f.RasterXSize)
        miny = maxy + (resy * reference_f.RasterYSize)

        quads_extent_dict = {}

        quad1_minx = str(minx)
        quad1_maxx = str(minx + ((maxx - minx) / 2))
        quad1_miny = str(miny + ((maxy - miny) / 2))
        quad1_maxy = str(maxy)
        quads_extent_dict[1] = " ".join([quad1_minx, quad1_miny, quad1_maxx, quad1_maxy])

        quad2_minx = str(quad1_maxx)
        quad2_maxx = str(maxx)
        quad2_miny = str(quad1_miny)
        quad2_maxy = str(maxy)

        quads_extent_dict[2] = " ".join([quad2_minx, quad2_miny, quad2_maxx, quad2_maxy])

        quad3_minx = str(minx)
        quad3_maxx = str(quad1_maxx)
        quad3_miny = str(miny)
        quad3_maxy = str(quad1_miny)

        quads_extent_dict[3] = " ".join([quad3_minx, quad3_miny, quad3_maxx, quad3_maxy])

        quad4_minx = str(quad1_maxx)
        quad4_maxx = str(maxx)
        quad4_miny = str(miny)
        quad4_maxy = str(quad1_miny)

        quads_extent_dict[4] = " ".join([quad4_minx, quad4_miny, quad4_maxx, quad4_maxy])

        localLog.debug("Clipping Quads for %s" % raster)
        for i in range(1, 5):
            #print("Starting on quad %d" % i)
            quad_name = "quad" + str(i)
            quad_dir = os.path.join(outdir, quad_name)
            
            utils.useDirectory(quad_dir)

            oname = os.path.splitext(os.path.basename(raster))[0] + "_" + quad_name + ".tif"
            opath = os.path.join(quad_dir, oname)

            if not os.path.exists(opath) or overwrite:
                ouput_options = "-overwrite -t_srs %s -tr %s %s -te_srs %s -te %s" % (
                    proj, resx, resy, proj, quads_extent_dict[i])

                localLog.info("Executing gdal_warp operation on %s with extent %s" % (raster, quads_extent_dict[i]))
                gdal.Warp(opath, raster, options=ouput_options)

    return outdir


def specifyTrainingRasters(alist, to_remove):
    keepers = alist[:]
    for ras in alist:
        for string in to_remove:
            if string in ras:
                keepers.remove(ras)

    keepers = sorted(keepers)
    return keepers


def pickSampleRaster(directory, fileName):
    """ given a directory returns the path of the first file matching the first file matching fileName """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == fileName:
                fpath = os.path.join(root, file)
                print("Sample raster %s" % fpath)
                return fpath

    if "fpath" not in locals():
        localLog.error("Unable to find sample raster matching %s in directory %s. Aborting." % (fileName, directory))
        raise ValueError


def extractValuesToVBPoints(watersheds_shp, vb_classification_pnts, watershedsDir):
    # Extract raster values to training points
    watersheds_df = gpd.read_file(watersheds_shp)
    class_points_df = gpd.read_file(vb_classification_pnts)

    if not watersheds_df.crs == class_points_df.crs:
        class_points_df.to_crs(watersheds_df.crs, inplace=True)

    vbpoints_ras_extract = gpd.sjoin(class_points_df, watersheds_df, op='within')

    sample_raster = pickSampleRaster(watershedsDir, "elev_cm.tif")
    raster_proj = getRasterProj4(sample_raster)

    localLog.debug("Reprojecting training points to coordinate system of rasters...")
    vbpoints_ras_extract.to_crs(raster_proj, inplace=True)
    print(vbpoints_ras_extract.crs)

    for watershed, group in vbpoints_ras_extract.groupby("HUC4"):
        localLog.info("\nStarting extraction on points in watershed %s" % watershed)

        # FIND RELEVANT WATERSHED DIRECTORY
        # TODO This is a redundant workflow in this and the VBET Script
        for wdir in os.listdir(watershedsDir):
            if watershed in wdir:
                localLog.debug("--- BEGINNING ON WATERSHED %s ---" % wdir)
                w_dir = os.path.join(watershedsDir, wdir)
                for subdir in os.listdir(w_dir):
                    if "Rasters" in subdir:
                        rasters_dir = os.path.join(w_dir, subdir)
                        predictors_dir = os.path.join(rasters_dir, "Predictors")
                        print("Using predictors_dir ", predictors_dir)
                        break

                    if "GDB" in subdir:
                        geodatabase = os.path.join(w_dir, subdir)

        createFlowlineBufferRaster(watersheds_dir)# overwrite=True)
        # simple check to make sure that predictors have been made.
        # TODO - initiate calculation if not
        elev_raster = os.path.join(predictors_dir, "elev_meters.tif")
        if not os.path.exists(elev_raster):
            localLog.ERROR("PROBLEM - %s doesn't exist in directory %s" % ("elev_meters.tif", predictors_dir))
            raise Exception    
            
        float32_raster_paths = []
        raster_names = []
        for root, dirs, files in os.walk(predictors_dir):
            for file in files:
                if file.endswith(".tif") or file.endswith(".img"):
                    if file.lower() != "elev_meters.tif":
                        raster_names.append(file[:-4])  # append to list without file extension
                        float32_raster_paths.append(os.path.join(root, file))

        rasters = [elev_raster] + float32_raster_paths
        raster_names = ["elev_meters"] + raster_names

        print("Extracting point values for rasters: ", raster_names)

        for name in raster_names:
            vbpoints_ras_extract[name] = np.NaN
            
        # Build VRT to makes extraction easier/simpler. Can't include elev_meters because different data type
        localLog.debug("Building VRT of FLOAT32 Rasters...")
        vrt_of_rasters = os.path.join(predictors_dir, "float32_predictors.vrt")
        build_vrt = "gdalbuildvrt -overwrite -separate %s %s" % (vrt_of_rasters, '"' + '" "'.join(float32_raster_paths) +'"')
        #print("Executing command to system:\n %s\n" % build_vrt)
        os.system(build_vrt)

        def get_values(geom):
            
            x = geom.centroid.x
            y = geom.centroid.y

            values = []

            for val in elev_ras.sample([(x, y)]):
                values += np.ndarray.tolist(val)
            for val in float32_ras.sample([(x, y)]):
                values += np.ndarray.tolist(val)

            return pd.Series(values, index=raster_names)
            
            
        with rio.open(elev_raster) as elev_ras:
            with rio.open(vrt_of_rasters) as float32_ras:
                vbpoints_ras_extract.loc[vbpoints_ras_extract.HUC4 == watershed, raster_names] = \
                    vbpoints_ras_extract.loc[vbpoints_ras_extract.HUC4 == watershed, "geometry"].apply(get_values)

        #print(vbpoints_ras_extract.loc[vbpoints_ras_extract.HUC4 == watershed,])
        #break

    return {"points": vbpoints_ras_extract, "raster_paths": rasters, "raster_names": raster_names}


def regressValleyBottoms(vbpoints_ras_extract, rasters_to_use, rasters_to_not_use, vb_dir, watershedsDir, overwrite=False):
    localLog.debug("Beginning Regression Training")
    n_job = 2
    msl = 20
    print("rasters_to_use", rasters_to_use)
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    regressor = RandomForestRegressor(n_jobs=n_job, verbose=True, min_samples_leaf=msl)

    regressor.fit(vbpoints_ras_extract[rasters_to_use].dropna(),
                  vbpoints_ras_extract[rasters_to_use + ["VB"]].dropna()["VB"])

    # CREATE CLASSIFIED RASTERS FOR QUARTER QUADS USED IN TRAINING DATA FIRST
    for watershed, group in vbpoints_ras_extract.groupby("HUC4"):

        # FIND RELEVANT WATERSHED DIRECTORY
        ## TODO, this is a redundant workflow in this file and in the VBET process. Create function to replace
        for wdir in os.listdir(watershedsDir):
            if watershed in wdir:
                localLog.debug("--- BEGINNING ON WATERSHED %s ---" % wdir)
                watershedDir = os.path.join(watershedsDir, wdir)
                for subdir in os.listdir(watershedDir):
                    if "Rasters" in subdir:
                        rasters_dir = os.path.join(watershedDir, subdir)
                    # if "GDB" in subdir:
                    #    geodatabase = os.path.join(watershedDir, subdir)

                    predictors_dir = os.path.join(rasters_dir, "Predictors")

                break

        # This step divide the watershed predictors into 4 quadrants and return the location of the folder
        quadsDir = rasterSubDivide(predictors_dir, overwrite=False)

        # Set the output directory to write prediction rasters to
        outquad_preds_dir = os.path.join(watershedDir, "RSAC_temp")
        utils.useDirectory(outquad_preds_dir)

        for subdir in os.listdir(quadsDir):
            
            dirpath = os.path.join(quadsDir, subdir)
            raster_paths = sorted(getRasterNamesList(dirpath)[1])

            qquad_rasters_to_use = sorted(specifyTrainingRasters(raster_paths, rasters_to_not_use))
            print("qquad_rasters_to_use: ", qquad_rasters_to_use)
            
            for i in range(len(rasters_to_use)):
                localLog.debug(i, rasters_to_use[i], os.path.basename(raster_paths[i]))
            
            if len(rasters_to_use) != len(rasters_to_use):
                raise Exception

            localLog.debug("Starting on directory : %s" % dirpath)

            modeltype = "RandomForestsReg"
            output_fname = "VB_" + watershed + "_" + subdir + "_" + modeltype + ".tif"
            loc_classified_file = os.path.join(outquad_preds_dir, output_fname)

            classified_File = createClassifiedFile(qquad_rasters_to_use, regressor, loc_classified_file, overwrite=True)

        # NEED TO MERGE QUADS OF WATERSHED BACK TO ONE THE WATERSHED
        watershed_rsac_name = "VB_" + watershed + "_" + modeltype + ".tif"
        watershed_rsac_path = os.path.join(watershedDir, watershed_rsac_name)

        if not os.path.exists(watershed_rsac_path) or overwrite:
            quadfiles = []
            for file in os.listdir(outquad_preds_dir):
                if modeltype in file and file.endswith(".tif"):
                    fpath = os.path.join(outquad_preds_dir, file)
                    quadfiles.append(fpath)

            utils.mergeRasters(quadfiles, watershed_rsac_path)

    # MERGE ALL WATERSHEDS TO ONE RASTER FOR WHOLE STATE
    state_rsac_name = "RSAC_ValleyBottoms.tif"
    state_rsac_path = os.path.join(vb_dir, state_rsac_name)

    if not os.path.exists(state_rsac_path) or overwrite:
        watershedfiles = []
        for w_dir in os.listdir(watershedsDir):
            watershedDir = os.path.join(watershedsDir, w_dir)
            for file in os.listdir(watershedDir):
                if modeltype in file and file.endswith(".tif"):
                    fpath = os.path.join(watershedDir, file)
                    watershedfiles.append(fpath)

        utils.mergeRasters(watershedfiles, state_rsac_path)


def valleyBottomRegression(watersheds_shp, vb_classification_pnts, vbdir, watersheds_dir, rasters_to_not_regress_with):
    extraction_variables = extractValuesToVBPoints(watersheds_shp, vb_classification_pnts, watersheds_dir)

    vbpoints_raster_values = extraction_variables["points"]
    vbpoints_raster_values.to_file(r"M:\Data\ValleyBottoms\Watersheds\trainingPoints_extracts.shp")
    #rasters = extraction_variables["raster_paths"]
    raster_names = extraction_variables["raster_names"]

    print("raster_names: ", raster_names)

    # remove the rasters that were reated in prep, but will not actually be used in regression
    #rasters_to_not_use = ["TPI_10", "TPI_20", "TPI_30", "Euc_times_Slope", "Slope_Degrees"]

    rasters_to_regress_with = specifyTrainingRasters(raster_names, rasters_to_not_regress_with)

    localLog.debug("Rasters to use in regression: ", rasters_to_regress_with)

    # Only some training data will be used. Those marked with a 1 in 'Use' column
    vbpoints_raster_values = vbpoints_raster_values[vbpoints_raster_values.Use == 1]

    regressValleyBottoms(vbpoints_raster_values, rasters_to_regress_with, rasters_to_not_regress_with, vbdir, watersheds_dir)


def getRasterProj4(raster):
    print("Getting projection information for %s" % raster)
    """ Function returns the projection of the input raster in proj4"""
    fac = gdal.Open(raster)

    ras_proj = fac.GetProjection()
    spatialRef = osr.SpatialReference()

    osr.UseExceptions()
    # Apparently osr has difficulties identifying albers projections
    prjText = ras_proj.replace('"Albers"', '"Albers_Conic_Equal_Area"')
    spatialRef.ImportFromWkt(prjText)
    ras_proj_proj4 = spatialRef.ExportToProj4()
    return ras_proj_proj4


def getResAndExtent(raster_file):
    """ RETURN THE RESOLUTION AND EXTENT OF THE RASTER AS A LIST [resx, resy, xmin, ymin, xmax, ymax]"""
    with rio.open(raster_file) as ras:
        ymax = ras.profile['transform'][5]
        xmin = ras.profile['transform'][2]
        height = ras.profile['height']
        width = ras.profile['width']
        resx = ras.profile['transform'][0]
        resy = ras.profile['transform'][4]
        ymin = ymax + (height * resy)
        xmax = xmin + (width * resx)

        return [abs(resx), abs(resy), str(xmin), str(ymin), str(xmax), str(ymax)]


def createFlowlineBufferRaster(watershedsDir, overwrite=False, cleanup=False):
    """ Creates a single output valley bottom using the VBET methodology. First iterates watersheds directory
    for each HUC4 watersheds and """

    # The final output of the script
    #vbet_allwatersheds = os.path.join(indir, "VBET_ValleyBottoms.tif")

    # Watershed Size Column Name
    watershedsize_col = "TotDASqKm"

    for w_dir in os.listdir(watershedsDir):
        localLog.info("--- BEGINNING ON WATERSHED %s ---" % w_dir)
        watershed_dir = os.path.join(watershedsDir, w_dir)
        for subdir in os.listdir(watershed_dir):
            if "Rasters" in subdir:
                rasters_dir = os.path.join(watershed_dir, subdir)
            if "GDB" in subdir:
                geodatabase = os.path.join(watershed_dir, subdir)

        fac_raster_loc = os.path.join(rasters_dir, "fac.tif")
        preds_dir = os.path.join(rasters_dir, "Predictors")
        intermediate_preds_dir = os.path.join(rasters_dir, "RSAC_Intermediates")
        dem_ras = os.path.join(preds_dir, "elev_meters.tif")
        slope_ras = os.path.join(intermediate_preds_dir, "Slope.tif")

        outbuffer_ras = os.path.join(preds_dir, "WatershedBufferSize.tif")

        raster_crs = getRasterProj4(dem_ras)

        if not os.path.exists(slope_ras) or not os.path.exists(dem_ras):
            localLog.error(
                "Slope raster does not exist. Run RSAC preprocessing script from ArcGIS python environment.")
            raise Exception

        if not os.path.exists(outbuffer_ras) or overwrite:
            localLog.info("%s doesn't exist. Beginning creation..." % outbuffer_ras)
            flowlines_vector = os.path.join(watershed_dir, "NHD_Flowlines_buffered.shp")
            if not os.path.exists(flowlines_vector) or overwrite:
                localLog.debug("Reading in NHD flowlines feature class from geodatabase...")
                flowlines = gpd.GeoDataFrame.from_file(geodatabase, layer='NHDFlowline')
                f_crs = flowlines.crs

                # GET VAA TABLE AND JOIN TO FEATURE CLASS FOR WATERSHED SIZE
                flowlines_vaa = gpd.GeoDataFrame.from_file(geodatabase, layer='NHDPlusFlowlineVAA')

                flowlines_merge = flowlines.merge(flowlines_vaa, on='NHDPlusID')

                # Merge renames geometry to geometry_x. Fix
                flowlines_merge['geometry'] = flowlines_merge['geometry_x']

                # merge is a pandas operation, re-read in as a geodataframe
                flowlines_merge = gpd.GeoDataFrame(flowlines_merge, crs=f_crs, geometry="geometry")

                flowlines_merge.to_crs(raster_crs, inplace=True)

                localLog.debug("Reprojecting flowlines dataframe to FAC raster projection...")

                # TODO - select only flowlines which are true in-ground streams. Do not include canals, culverts, etc
                # pipline underground  - FCODE 42803, 42804, 42807, 42808, 42812...

                # Cleanup flowlines table by removing all columns not geometry
                drop_columns = flowlines_merge.columns.tolist()
                drop_columns.remove('geometry')
                drop_columns.remove(watershedsize_col)
                flowlines_merge.drop(drop_columns, axis=1, inplace=True)

                # get resolution
                with rio.open(fac_raster_loc) as ras:
                    res_x, res_y = ras.res

                # pixel X number in km
                ratio_x = 1000 / res_x
                ratio_y = 1000 / res_y

                # convert watershed size from km to m
                flowlines_merge["TotDASq_m"] = flowlines_merge[watershedsize_col] * ratio_x * ratio_y

                def calculateBufferSize(da):
                    if da <= 1:
                        buff_size = 1
                    else:
                        buff_size = math.sqrt(da) / (math.log(da, 10) * (4 / 3))

                    return buff_size

                def bufferLines(row):
                    geom = row.geometry
                    buffersize = row.BufferSize
                    # fac = row[watershedsize_col]

                    # log of 1 is 0, can't divide by zero. Also, a Flow accumulation value of 1 or zero is a misread, essentially minimum buffer size
                    # if fac <= 1:
                    #    fac = 2

                    """Simple equation which correlates the flow accumulation values (fac), e.g. watershed size,
                    to the appropriate valley bottom buffer"""
                    # buffersize = math.sqrt(fac) / (math.log(fac, 10) * (4 / 3))

                    return geom.buffer(buffersize)

                flowlines_merge["BufferSize"] = flowlines_merge["TotDASq_m"].apply(calculateBufferSize)

                # Buffer each flowline by its watershed size
                flowlines_merge["geometry"] = flowlines_merge.apply(bufferLines, axis=1)

                # Write out to shapefile
                localLog.debug("Writing out flowline buffers to file: %s" % flowlines_vector)
                #print("Flowline dtypes:\n {}".format(flowlines.dtypes))

                flowlines_merge.to_file(flowlines_vector)
            else:
                flowlines_merge = gpd.read_file(flowlines_vector)

            with rio.open(dem_ras) as ras:
                # copy and update the metadata from the input raster for the output
                meta = ras.meta.copy()
                # meta.update(compress='lzw')
                shape = ras.shape
                inras_crs = ras.crs.from_string(raster_crs)

            meta.update(
                dtype=np.float32,
                nodata=-9999,
                crs=inras_crs
            )

            # create empty blank array
            out_arr = np.full(shape, -9999.0, dtype=np.float32)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            feat_shapes = ((geom, value) for geom, value in zip(flowlines_merge.geometry, flowlines_merge.BufferSize))
            # write generator to array
            burned_array = features.rasterize(shapes=feat_shapes, fill=-9999, out=out_arr,
                                                  transform=meta['transform'], dtype=np.float32)
            # write array to file
            print("Writing %s to file" % outbuffer_ras)
            with rio.open(outbuffer_ras, 'w', **meta) as out_ras:
                out_ras.write_band(1, burned_array.astype(np.float32))

            print("FINISHED")

            if cleanup:
                os.remove(flowlines_vector)


if __name__ == "__main__":
    import os
    import pandas as pd
    import geopandas as gpd
    import gdal, osr
    import rasterio as rio
    from rasterio import features
    import numpy as np
    from datetime import datetime
    from sklearn.ensemble import RandomForestRegressor
    import logging as localLog
    import math

    import Utilities as utils

    localLog.basicConfig(level=localLog.INFO)
    
    valley_bottom_dir = os.path.abspath(r"M:\Data\ValleyBottoms")
    watersheds_dir = os.path.join(valley_bottom_dir, "Watersheds")

    vb_classification_pnts = os.path.join(r"M:\Data\inital_model_inputs", "VM_TrainingData_20180619.shp")
    watersheds_shp = os.path.join(r"M:\Data\inital_model_inputs", "WBDHU4_Arizona.shp")

    rasters_not_used = ["TPI_10", "TPI_20", "TPI_30", "Euc_times_Slope", "Slope_Degrees"]

    #createFlowlineBufferRaster(watersheds_dir, overwrite=True)

    valleyBottomRegression(watersheds_shp, vb_classification_pnts, valley_bottom_dir, watersheds_dir, rasters_not_used)
    
    localLog.info("Finished creating valley bottoms from regression")
else:
    import os
    import pandas as pd
    import geopandas as gpd
    import gdal, osr
    import rasterio as rio
    from rasterio import features
    import numpy as np
    from datetime import datetime
    from sklearn.ensemble import RandomForestRegressor
    import logging as localLog

    import Utilities as utils
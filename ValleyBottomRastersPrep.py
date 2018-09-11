# Raster variable preparation for valley bottom models. Derived and primarily used for the RSAC Model

def watershed_vb_prep(watershed_code, vb_dir, raster_folder, flow_threshold, overwrite=False):
    """
    Function takes an input of prepared NHD-HR raster data downloaded to the raster_folder directory and
    generate Valley Bottom training variables used in the RSAC regression model.

    The following raster variables are produced:
    - A DEM in meters (NHD download is in cm)
    - Flow accumulation (direct copy of NHD download)
    - Flow Direction
    - Drainage network
    - Slope in radians (smoothed)
    - Height above channel
    - Euclidean distance from channel
    - Euclidean distance x slope
    - Topographic Position Index (TPI) at 10m, 20m, & 30m radii)
    - Compound Topographic Wetness Index
    """
    print("Beginning raster preparation in watershed raster folder: \t%s" % raster_folder)
    ws_start = datetime.datetime.now()

    if not os.path.exists(raster_folder):
        print("Unable to find %s. Run getNHDData function from Utilities.py with %s" % (raster_folder, vb_dir))


    # Setup the directories and paths
    intermediate_folder = os.path.join(raster_folder, 'RSAC_Intermediates')
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)

    predictors_folder = os.path.join(raster_folder, "Predictors")
    if not os.path.exists(predictors_folder):
        os.makedirs(predictors_folder)

    # List of predictors available that will be filled
    predictors = []

    # SET FLOW ACCUMULATION RASTER NHD DEM RASTER, NEW DEM RASTER
    dem_flow_acc = Raster(os.path.join(raster_folder, "fac.tif"))
    nhd_swnet = Raster(os.path.join(raster_folder, "swnet.tif"))
    dem_cm = Raster(os.path.join(raster_folder, "elev_cm.tif"))
    dem_meters = os.path.join(predictors_folder, "elev_meters.tif")


    out_int_base = os.path.join(intermediate_folder, os.path.splitext(os.path.basename(dem_meters))[0])


    # CREATE THE DRAINAGE NETWORK BY THRESHOLDING THE FLOW ACCUMULATION LAYER
    # fac.tif is created from hydrodem.tif, but hydrodem has streamlines burned too deep for
    # good slope calculation. So, recreate hydrodem using streamlines
    dem_drainage_network = out_int_base + '_Drainage_Network_' + str(flow_threshold) + '.tif'

    if not os.path.exists(dem_drainage_network) or overwrite:
        print('\tThresholding the DEM drainage network from %s...' % dem_flow_acc)
        outCon = Con(dem_flow_acc >= flow_threshold, 1, 0)

        # ADD WATERBODIES FROM NHD FLOW NETWORK (VALUE == 2)
        #waterbodies = Con(nhd_swnet >= 1, 1, 0)

        new_swnet = Con(nhd_swnet >= 1, 1, Con(outCon == 1, 1, 254))

        new_swnet.save(dem_drainage_network)
        del outCon
        # del waterbodies

    else:
        new_swnet = Raster(dem_drainage_network)

    # for testing
    #dem_meters_temp = os.path.join(predictors_folder, "elev_meters_convert.tif")
    #if not os.path.exists(dem_meters_temp) or overwrite:
        #print("\tConverting DEM %s to meters..." % dem_cm)
        dem_meters_convert = Divide(dem_cm, 100)
        #dem_meters_convert.save(dem_meters_temp)
    #else:
    #   dem_meters_convert = Raster(dem_meters_temp)

    #dem_low = dem_meters_convert - 1

    if not os.path.exists(dem_meters) or overwrite:
        print("\tConverting DEM %s to meters..." % dem_cm)
        dem_meters_convert = Divide(dem_cm, 100)

        dem_low = dem_meters_convert - 1

        print("\tBurning in flowline network...")

        # SET noData VALUES TO 0
        new_swnet = Con(IsNull(nhd_swnet), 0, nhd_swnet)
        dem_hydro = Con(new_swnet == 1, dem_low, dem_meters_convert)
        #dem_hydro = dem_meters_convert - new_swnet

        dem_hydro.save(dem_meters)
        del dem_meters_convert
        del dem_hydro
        del new_swnet


    # Find some basic info about the base dem for use in subsequent steps
    info = arcpy.Describe(dem_meters)
    res = info.children[0].meanCellHeight

    # All subsequent variables are predictor variables

    # COMPUTE SLOPE IN RADIANS (PERCENT/100) AND SMOOTH IT USING A 6X6 CIRCULAR KERNEL TO REDUCE ARTIFACTS
    # COMPUTE SLOPE IN DEGREES AND SMOOTH IT USING A 6X6 CIRCULAR KERNEL TO REDUCE ARTIFACTS
    slope_degrees = os.path.join(intermediate_folder, "Slope.tif")
    if not os.path.exists(slope_degrees):
        print('\tCalculating degree slope from %s...' % dem_meters)
        slpdeg = Slope(dem_meters, z_factor=1)#, "PERCENT_RISE") / 100.0
        slpdeg.save(slope_degrees)
    else:
        slpdeg = Raster(slope_degrees)

    slope_degrees_smooth = os.path.join(predictors_folder, "Slope_Degrees.tif")
    predictors.append(slope_degrees_smooth)
    if not os.path.exists(slope_degrees_smooth) or overwrite:
        neighborhood = NbrCircle(3, "CELL")
        slpDegSmth = FocalStatistics(slpdeg, neighborhood, "MEAN")
        slpDegSmth.save(slope_degrees_smooth)
        del slpdeg
        del slpDegSmth

    # COMPUTE HEIGHT ABOVE CHANNEL (ASSUMED TO BE IN METERS)
    hac = os.path.join(predictors_folder, 'Height_Above_Channel.tif')
    predictors.append(hac)
    if not os.path.exists(hac) or overwrite:
        print('\tCalculating the height above the channel...')
        outHAC = CostDistance(dem_drainage_network, slope_degrees_smooth)
        outHAC.save(hac)
        del outHAC

    # COMPUTE EUCLIDEAN DISTANCE FROM CHANNEL
    euc = os.path.join(predictors_folder, 'Euclidean_Distance_from_Channel.tif')
    predictors.append(euc)
    if not os.path.exists(euc) or overwrite:
        print('\tCalculating the euclidean distance from channel...')
        outEuc = EucDistance(dem_drainage_network, cell_size=res)
        outEuc.save(euc)
        del outEuc

    # COMPUTE THE PRODUCT OF ECULIDEAN DISTANCE AND SLOPE (SLIGHTLY DIFFERENCE FROM THE HEIGHT ABOVE CHANNEL)
    eucxslope = os.path.join(predictors_folder, 'Euc_times_Slope.tif')
    predictors.append(eucxslope)
    if not os.path.exists(eucxslope) or overwrite:
        print('\tMultiplying the euclidean distance by the slope...')
        outEucxSlope = Times(euc, slope_degrees_smooth)
        outEucxSlope.save(eucxslope)
        del outEucxSlope

    # Computes the topographic position index (TPI)
    # Difference in elevation between a pixel and the average of its neighborhood
    # Currently uses a circular kernel of varying diameters
    # The z suffix indicates that it is the z score of the elevation within a given neighborhood
    tpis = [10] #, 20, 30]  # ,40, 60]
    for tpi in tpis:
        tpi_out = os.path.join(predictors_folder, 'TPI_' + str(tpi) + '.tif')
        tpi_outz = os.path.join(predictors_folder, 'TPI_' + str(tpi) + 'z.tif')
        predictors.append(tpi_out)
        predictors.append(tpi_outz)
        if not os.path.exists(tpi_out) or not os.path.exists(tpi_outz) or overwrite:
            neighborhood = NbrCircle(tpi / 2, "CELL")
            print('\tComputing the topographic position index neighborhood %d...' % tpi)
            mean = FocalStatistics(dem_meters, neighborhood, "MEAN")
            std = FocalStatistics(dem_meters, neighborhood, "STD")
            tpir = dem_meters - mean
            tpir.save(tpi_out)
            del tpir
            tpizr = (dem_meters - mean) / std
            tpizr.save(tpi_outz)
            del mean
            del std
            del tpizr

    # COMPUTE THE COMPOUND TOPOGRAPHIC WETNESS INDEX (CTWI) AND MOOTH USING A 14X14 CIRCULAR KERNEL
    ctwi = os.path.join(predictors_folder, "DEM_CTWI.tif")
    predictors.append(ctwi)
    if not os.path.exists(ctwi) or overwrite:
        print('\tComputing the compound topographic wetness index...')
        #acc = Raster(dem_flow_acc)
        slp = (Raster(slope_degrees_smooth) * 1.570796) / 90
        tan_slp = Con(slp>0, Tan(slp), 0.001)
        dem_flow_acc_scaled = (dem_flow_acc + 1) * (res * res)
        twi = Ln(dem_flow_acc_scaled/tan_slp)
        neighborhood = NbrCircle(7, "CELL")

        ctwi_smooth = FocalStatistics(twi, neighborhood, "MEAN")
        ctwi_smooth.save(ctwi)

        del slp
        del ctwi_smooth

    del dem_flow_acc
    del nhd_swnet
    del dem_cm

    print('\tValley bottom prep is complete for watershed %s' % watershed_code)
    print("\tElapsed: %s" % str(datetime.datetime.now() - ws_start))

    return predictors


def vb_prep(wtrshs_folder, flow_initiation_threshold=2000, overwrite=False):
    watersheds_folder = getParentDir(wtrshs_folder)
    vbdir = getParentDir(watersheds_folder)

    print("Beginning Valley Bottom prep for watersheds in watershed directory %s " % wtrshs_folder)
    for item in os.listdir(wtrshs_folder):
        # Check to make sure it is a directory and that it's 4 characters long (HUC4 code)
        ipath = os.path.join(wtrshs_folder, item)
        if os.path.isdir(ipath) and len(item) == 4:
            watershed_num = item
            watershed_dir = os.path.join(wtrshs_folder, watershed_num)
            for subdir in os.listdir(watershed_dir):
                if "Rasters" in subdir:
                    wtrshd_raster_folder = os.path.join(watershed_dir, subdir)
                    watershed_vb_prep(watershed_num, vbdir, wtrshd_raster_folder, flow_threshold=flow_initiation_threshold, overwrite=overwrite)


def getParentDir(p):
    return os.path.abspath(os.path.join(p, os.pardir))


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


if __name__ == '__main__':
    import os
    import datetime
    from math import tan
    import logging as localLog

    try:
        import arcpy
        from arcpy.sa import *
        arcpy.env.overwriteOutput = True
    except ImportError as e:
        raise Exception("Unable to import arcpy module. Run this script from an environment with access to arcpy.")

    vb_watersheds_dir = os.path.abspath(r"M:\Data\ValleyBottoms\Watersheds")

    overwrite = True

    flow_acc_threshold = 2000

    vb_prep(vb_watersheds_dir, flow_initiation_threshold=flow_acc_threshold, overwrite=False)
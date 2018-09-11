
# VBET VAllEY BOTTOM MODEL
#     The final output of the script is placed in nhd_dir as "VBET_ValleyBottoms.tif"



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


def calculateGeom(row):
    """ Is passed a row containing flowline geometry. Finds the node
    in the geometry which is closest to center and then creates and 
    returns a 5m buffered polygon"""
    geom = row["geometry"]
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    num_nodes = len(geom.coords)
    if not num_nodes == 2:
        # IF LINESTRING IS NOT COMPOSED OF ONLY TWO NODES, GET THE NODE IN THE MIDDLE MOST OF THE LINE
        point = Point(geom.coords[int(num_nodes/2)])
        bufferSize = 10
    else:
        #print("Two point linestring...")
        # IF TWO POINT LINESTRING, SMALL STREAM ANYWAY. TAKE THE POINT WHICH IS AT THE END OF THE LINE
        point = Point(geom.coords[-1])
        bufferSize = 5
    
    poly = point.buffer(bufferSize)
        
    #points = createPoints(poly, fac_raster_loc)
    
    return poly


def getSnappedPixelLocation(geom_x, geom_y, ras_aff):
    #print("GEOM_X: ", geom_x, "GEOM_Y: ", geom_y)
    """ Returns set of upper-right snapped pixel locations in set as (x, y)"""
    pix_xsize = ras_aff.a
    pix_ysize = ras_aff.e
    #print(pix_xsize, pix_ysize)

    # get pixel coordinates of the geometry's bounding box
    xvals = sorted([geom_x, ras_aff.c])
    yvals = sorted([geom_y, ras_aff.f])
    #print("XVALS: ", xvals)

    diffx = xvals[1] - xvals[0]
    diffy = yvals[1] - yvals[0]
    #print("DIFFS: ", diffx, diffy)

    pixel_xdiff = float("{0:.11f}".format( diffx % pix_xsize ))  # get modulo pixel difference to float precision of 11 decimals
    pixel_ydiff = float("{0:.11f}".format( diffy % pix_ysize ))  # get modulo pixel difference to float precision of 11 decimals
    #print("PIXEL DIFF: ", pixel_xdiff, pixel_ydiff)

    #snapped pixel locations
    if pixel_xdiff > pix_xsize / 2:
        snapped_ulx = geom_x + (pix_xsize - pixel_xdiff)
    else:
        snapped_ulx = geom_x - pixel_xdiff
   
    if abs(pixel_ydiff) > abs(pix_ysize / 2):
        snapped_uly = geom_y + (abs(pix_ysize) + pixel_ydiff)
    else:
        snapped_uly = geom_y - abs(pixel_ydiff)
            
    if snapped_ulx % pix_xsize != ras_aff.c % pix_xsize:
        print(snapped_ulx % pix_xsize)
        raise ValueError("BAD PIXEL VALUE FOR ULX - ", snapped_ulx)
    if snapped_uly % pix_ysize != ras_aff.f % pix_ysize:
        print(snapped_uly % pix_ysize)
        raise ValueError("BAD PIXEL VALUE FOR ULY - ", snapped_uly)
    
    return {"x": snapped_ulx, "y": snapped_uly}


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


def getRasterTransform(rasterLoc):
    with rio.open(rasterLoc) as raster:
        t = raster.affine

    return t


def mergeFlowlineSizeRasters(rasters, out_raster):
    localLog.info("Merging raster sizes to final at %s" % out_raster)
    arrays = []
    for i in range(len(rasters)):
        with rio.open(rasters[i]) as ras:
            kwargs = ras.profile
            ras_array = ras.read().astype(float)
            arrays.append(ras_array)

    max_array = arrays[0]
    for i in range(len(arrays) - 1):
        max_array = np.maximum(arrays[i], arrays[i + 1])

    kwargs['dtype'] = max_array.dtype

    with rio.open(out_raster, 'w', **kwargs) as dst:
        dst.write(max_array)


def rasterizeBufferedFlowlines(features_df, attrib, sample_raster, out_raster, overwrite=False):
    if not os.path.exists(out_raster) or overwrite:
        print("\tRasterizing features...")
        raster_crs = getRasterProj4(sample_raster)

        with rio.open(sample_raster) as ras:
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
        feat_shapes = ((geom, value) for geom, value in
                       zip(features_df.geometry, features_df[attrib]))
        # write generator to array
        burned_array = features.rasterize(shapes=feat_shapes, fill=-9999, out=out_arr,
                                          transform=meta['transform'], dtype=np.float32)
        # write array to file
        print("Writing %s to file" % out_raster)
        with rio.open(out_raster, 'w', **meta) as out_ras:
            out_ras.write_band(1, burned_array.astype(np.float32))

        print("Finished buffer")
        return out_raster
    else:
        return out_raster


def createVBETValleyBottom(indir, watershedsDir, slope_thresh_dict, drainage_thresh_dict, overwrite=False, cleanup=False):
    """ Creates a single output valley bottom using the VBET methodology. First iterates watersheds directory 
    for each HUC4 watersheds and """

    # The final output of the script
    vbet_allwatersheds = os.path.join(indir, "NHD_ValleyBottoms.tif")

    # Watershed Size Column Name
    watershed_col = "TotDASqKm"

    if not os.path.exists(vbet_allwatersheds) or overwrite:
        localLog.info("\nValley Bottom Raster based on VBET methodology doesn't exist. Beginning creation.\n")
        # Final output file doesn't exist, begin creation

        flow_acc_thresh = 2000  # minimum flow accumulation size to identify stream
        # ValleyBottomRastersPrep.vb_prep(watershedsDir, flow_initiation_threshold=flow_acc_thresh)

        # Need to divide by 1000 because of the PercentRise calculation used in Esri's Slope Determination. Just a component of predictor variables.
        #lrgSlopeThresh = lrgSlopeThresh / 1000
        #medSlopeThresh = medSlopeThresh / 1000
        #smSlopeThresh = smSlopeThresh / 1000

        """ Creates a single output valley bottom using the VBET methodology. First iterates watersheds directory
            for each HUC4 watersheds and """

        # The final output of the script
        # vbet_allwatersheds = os.path.join(indir, "VBET_ValleyBottoms.tif")

        # Watershed Size Column Name
        watershedsize_col = "TotDASqKm"
        out_raster_name = "NHD_Buffer_SlopeClip.tif"

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
            outbufferclip_ras = os.path.join(watershed_dir, out_raster_name)

            raster_crs = getRasterProj4(dem_ras)

            if not os.path.exists(slope_ras) or not os.path.exists(dem_ras):
                localLog.error(
                    "Slope raster does not exist. Run RSAC preprocessing script from ArcGIS python environment.")
                raise Exception

            if not os.path.exists(outbuffer_ras) or overwrite:
                localLog.info("%s doesn't exist. Beginning creation..." % outbuffer_ras)
                flowlines_vector = os.path.join(watershed_dir, "NHD_Flowlines_buffered.shp")
                if not os.path.exists(flowlines_vector):
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
                    # print("Flowline dtypes:\n {}".format(flowlines.dtypes))

                    flowlines_merge.to_file(flowlines_vector)
                else:
                    print("reading file")
                    flowlines_merge = gpd.read_file(flowlines_vector)

                rasterizeBufferedFlowlines(flowlines_merge, "TotDASq_m", dem_ras, outbuffer_ras, overwrite=False)

            with rio.open(outbuffer_ras) as ob_ras:
                buffered_features_array = ob_ras.read().astype(rio.float32)

            localLog.debug("Reading in slope raster")
            with rio.open(slope_ras) as sloperas:
                kwargs = sloperas.profile
                slope_array = sloperas.read().astype(rio.float32)

            temp_dir = os.path.join(watershed_dir, "VBET_temp")
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)

            # set nodata value to zero
            #slope_array = np.where(slope_array == kwargs["nodata"], 0, slope_array)

            # clip slope array to only buffered features area
            slope_array = np.where(slope_array < 0, -9999, slope_array)

            localLog.info("Clipping flowline buffers by slope")
            small_watercourses = np.where(buffered_features_array <= drainage_thresh_dict["Low"], np.where(buffered_features_array >= 0, 1, 0), 0)
            small_clip = np.where(slope_array < slope_thresh_dict["Small"], small_watercourses, 0)

            medium_watercourses = np.where(buffered_features_array > drainage_thresh_dict["Low"], np.where(buffered_features_array < drainage_thresh_dict["High"], 1, 0), 0)
            medium_clip = np.where(slope_array < slope_thresh_dict["Medium"], medium_watercourses, 0)

            large_watercourses = np.where(buffered_features_array >= drainage_thresh_dict["High"], 1, 0)
            large_clip = np.where(slope_array < slope_thresh_dict["Large"], large_watercourses, 0)

            slope_clip = (small_clip + medium_clip + large_clip).astype(np.uint8)

            kwargs.update(
                dtype=np.uint8,
                nodata=0
            )

            localLog.info("Writing watershed valleybottoms raster to file")
            with rio.open(outbufferclip_ras, "w", **kwargs) as dst:
                dst.write(slope_clip)

        # Merge All Watershed VBs together
        files = []

        for w_dir in os.listdir(watersheds_dir):
            watershed_dir = os.path.join(watersheds_dir, w_dir)
            for file in os.listdir(watershed_dir):
                if file == out_raster_name:
                    fpath = os.path.join(watershed_dir, file)
                    files.append(fpath)

        utils.mergeRasters(files, vbet_allwatersheds)

        """
        localLog.info("Beginning merge of :\n\t%s" % ("\n\t".join(files)))
        sources = [rio.open(f) for f in files]
        merged_array, output_transform = merge_tool(sources)
    
        profile = sources[0].profile
        profile['transform'] = output_transform
        profile['height'] = merged_array.shape[1]
        profile['width'] = merged_array.shape[2]
        
        profile.update(dtype=np.float32)
    
        #print(profile)

        localLog.debug("Writing merge out...")
        with rio.open(vbet_allwatersheds, 'w', **profile) as dst:
            dst.write(merged_array.astype(np.float32))

            localLog.info("Finished merging files to %s" % vbet_allwatersheds)
        """
    else:
        localLog.info("%s Already exists and no overwrite set" % vbet_allwatersheds)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    # Import necessary modules
    from shapely.geometry import Point
    from shapely.ops import linemerge
    import rasterio as rio
    import pandas as pd
    import geopandas as gpd
    import gdal, osr
    import numpy as np
    import os, shutil
    import math
    from rasterio.merge import merge as merge_tool
    from rasterio import features
    import Utilities as utils
    import logging as localLog
    import ValleyBottomRastersPrep

    localLog.basicConfig(level=localLog.INFO)

    overwrite = True
    cleanup = False
    vb_dir = os.path.abspath(r"M:\Data\ValleyBottoms")

    # print("Using NHD directory %s" % nhd_dir)

    watersheds_dir = os.path.join(vb_dir, "Watersheds")

    """Large Slope Threshold: The value that represents the upper limit of slopes that will be included in the valley bottom
    for the 'large' portions of the network."""
    large_slope_thresh = 3.2
    """Medium Slope Threshold: The value that represents the upper limit of slopes that will be included in the valley bottom
     for the 'medium' portions of the network."""
    medium_slope_thresh = 4.5
    """Small Slope Threshold: The value that represents the upper limit of slopes that will be included in the valley bottoms
     for the "small" portions of the network."""
    small_slope_thresh = 22

    slope_thresholds = {"Small": small_slope_thresh, "Medium": medium_slope_thresh, "Large": large_slope_thresh}

    """High Drainage Area Threshold: The drainage area value in square meters. Streams whose upstream drainage area is greater
    than this value will be considered the "large" portion of the network, and whose maximum valley bottom width will be
    represented with the "Large Buffer Size" parameter."""
    high_drainage_area_thresh = 1000000  # (m2)
    """Low Drainage Area Threshold: The drainage area value in square meters. Streams whose upstream drainage area is less
    than this value will be considered the "small" portion of the network, and whose maximum valley bottom width will be
    represented with the "Small Buffer Size" parameter. Streams whose upstream drainage area is between the high and low
    drainage area thresholds will be considered the "medium" portion of the network and their maximum valley bottom width
    represented by the "Medium Buffer Width" parameter."""
    low_drainage_area_thresh = 40000  # (m2)

    drainage_area_thresh = {"High": high_drainage_area_thresh, "Low": low_drainage_area_thresh}

    createVBETValleyBottom(vb_dir, watersheds_dir, slope_thresholds, drainage_area_thresh,
                           overwrite=True, cleanup=False)

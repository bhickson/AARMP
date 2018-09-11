from osgeo.gdalnumeric import *
from datetime import datetime, timedelta
import os, zipfile, re
import rasterio as rio
import Utilities as utilities

prismDir = r"M:\Data\PRISM"
days = 7

def getDayOfYear(file):
    # PRISM DOWNLOADS ARE IN FORMAT PRISM_ppt_stable_4kmD2_20150501_bil.bil
    day_string = re.findall(r'\d+', file)[2]  #slips string in to list of numeric chunks (e.g. [4, 2, 20150501])
    dateobj = datetime.strptime(day_string, '%Y%m%d')
    doy = str(dateobj.timetuple().tm_yday)
    return doy


def getPRISMData_BULK(element, start, end):
    global prismDir

    start = datetime.strptime(start, '%Y%m%d')
    end = datetime.strptime(end, '%Y%m%d')

    prismaddr = r'http://services.nacse.org/prism/data/public/4km/'

    outdir = prismDir + "/" + element

    utilities.useDirectory(outdir)

    day = end
    while day >= start:
        daystring = day.strftime("%Y%m%d")

        newFileName = element + "_" + daystring + ".zip"

        downloadaddr = prismaddr + element + "/" + daystring
        print(downloadaddr)
        filepath = outdir + "/" + newFileName
        if not os.path.exists(filepath):
            # Download the file via wcs for the relevant data
            utilities.downloadData(filepath, downloadaddr)
            print("Finished Download:", day)

            try:
                zipfile.ZipFile(filepath).extractall(outdir)
                os.remove(filepath)
            except:
                print("Unable to unzip ", filepath)


        day -= timedelta(days=1)

    print("FINISHED ALL YEAR DOWNLOADS FOR", element)


def calcPrevDates(prev_dates, calc):
    for date in prev_dates:
        raster_path = day_files[date]

        with rio.open(raster_path) as raster:
            ras = raster.read(1).astype(float)
            kwargs = raster.meta
        if "sum_ras" not in locals():
            sum_ras = ras
        else:
            sum_ras += ras

    if calc == "sum":
        ras = sum_ras
    else:
        ras = sum_ras / len(prev_dates)

    return [ras, kwargs]

# from https://stackoverflow.com/a/1060330
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield end_date - timedelta(n + 1)


start_date = "20150501"
end_date = "20150930"

prism_variables = ["vpdmax", "vpdmin"]
prism_variables = ["tmax"]
for var in prism_variables:
    # ITERATES OVER VARIABLES IN VARIABLE LIST AND DOES A BULK DOWNLOAD
    #getPRISMData_BULK(var, start_date, end_date)

    day_files = {}
    var_dir = prismDir + "/" + var
    # ITERATE OVER DIRECTORY AND ADD A day number: raster path KEY:VALUE PAIR TO DICTIONARY
    for root, dirs, files in os.walk(var_dir):
        for file in files:
            if file.endswith(".bil"):
                day = re.findall(r'\d+', file)[2]
                day_files[day] = os.path.join(root,file)

    startdateobj = datetime.strptime(start_date, '%Y%m%d')
    enddateobj = datetime.strptime(end_date, '%Y%m%d')

    startdateobj += timedelta(days=days) # averaging previous days, can't go too far back

    for single_date in daterange(startdateobj, enddateobj):
        date = single_date.strftime("%Y%m%d")
        print("starting", date)
        prev_dates = []
        for i in range(days):
            former = single_date - timedelta(days=i)
            prev_dates.append(former.strftime("%Y%m%d"))

        # CREATES AN AVERAGE OR SUM VALUE RASTER FOR ALL RASTERS SPANNING THE PREVIOUS 7 (DEFAULT) DAYS
        if var == "ppt":  # if precip, get sum
            calc_type = "sum"
        else:
            calc_type = "mean"

        prev_raster_info = calcPrevDates(prev_dates, calc_type)

        prev_raster = prev_raster_info[0]

        kwargs = prev_raster_info[1]    # USE DEFAULT kwargs FROM FIRST RASTER
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')

        outdir = prismDir + "/" + var + "/" + var + "_" + str(days) + "day" + calc_type
        utilities.useDirectory(outdir)

        ofile = outdir + "/" + var + "_" + date + ".tif"
        with rio.open(ofile, 'w', **kwargs) as dst:
            dst.write_band(1, prev_raster.astype(rio.float32))

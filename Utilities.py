# UNIVERSAL UTILITIES

import urllib.request
import shutil
import os
import subprocess
import zipfile
import tarfile
import rasterio as rio
from rasterio.merge import merge as merge_tool
import numpy as np
from irods.session import iRODSSession
import fiona


def getParentDir(p):
    return os.path.abspath(os.path.join(p, os.pardir))


def downloadData(fname, down_addr):
    with urllib.request.urlopen(down_addr) as response, open(fname, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def useDirectory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Created output directory - ", dir)
    return dir


def unpackTar(path, outfile):
    path = os.path.abspath(path)

    print("FILE:", outfile)
    parentdir = getParentDir(path)
    outpath = os.path.join(parentdir,outfile)
    print("OUTPATH", outpath)
    useDirectory(outpath)
    tarfile.open(os.path.abspath(path)).extractall(outpath)


def unpack7z(path, outdir=None):
    """ Function build for windows machine. Requires that 7z is installed on windows machines. Linux users will need
    a different subprocess call """

    if outdir == None:
        # use parent directory
        outdir = getParentDir(path)

    subprocess.call(r'"C:\Program Files\7-Zip\7z.exe" x ' + os.path.abspath(path) + ' -o' + outdir + ' -aos')


def unzip(file):
    zip_ref = zipfile.ZipFile(file, 'r')
    outdir = getParentDir(file)
    zip_ref.extractall(outdir)
    zip_ref.close()


def installPackage(package):
    if package == "rios":
        furl = r"https://bitbucket.org/chchrsc/rios/downloads/rios-1.4.5.tar.gz"
    elif package == "fmask":
        furl = r"https://bitbucket.org/chchrsc/python-fmask/downloads/python-fmask-0.4.5.tar.gz"
    else:
        print("Unknown package. Exiting")

    tar_name = os.path.basename(furl)
    file_name = tar_name[:-7]
    print(file_name)
    if not os.path.exists(os.path.abspath(tar_name)):
        downloadData(tar_name, furl)

    print(os.path.abspath(file_name))
    unpackTar(os.path.abspath(tar_name), file_name)

    print("Installing package: ", package)
    os.chdir(os.path.join(file_name,file_name))  # move into rios-1.4.5 raw dir
    package_build = "python setup.py build"
    package_install = "python setup.py install"
    print("Building package...")
    os.system(package_build)
    print("\nInstalling package...")
    os.system(package_install)
    os.chdir("../..")  # switch back to  original dir
    print("Finished installing package", package)


def getNHDData(directory):
    """ Function uses the url of the NHD data on Amazon S3, Download to folder "Download" in the given directory
    and unpacks into "Watersheds" in the given directory"""

    url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlus/HU4/HighResolution/GDB/NHDPlus_H_HUCCODE_DATATYPE"

    useDirectory(directory)

    watersheds_dir = os.path.join(directory, "Watersheds")
    useDirectory(watersheds_dir)

    downloads_dir = os.path.join(directory, "Downloads")
    useDirectory(downloads_dir)

    # Hydrologic Unit Code (HUC) for HUs intersecting Arizona at the level 4
    # TODO - should these be passed to the function instead?
    hucs = ["1407", "1408", "1501", "1502", "1503", "1504", "1505", "1506", "1507", "1508"]
    for huc in hucs:
        print("Starting HUC", huc)
        huc_dir = os.path.join(watersheds_dir, huc)
        useDirectory(huc_dir)

        huc_url = url.replace("HUCCODE", huc)

        for dt in ["RASTER.7z", "GDB.zip"]:
            dt_url = huc_url[:].replace("DATATYPE", dt)

            fname = os.path.basename(dt_url)
            filepath = os.path.join(downloads_dir, fname)

            # Download the file
            if not os.path.exists(filepath):
                print("Getting", fname)
                downloadData(filepath, dt_url)
            else:
                print("Found existing file here %s" % filepath)

            unpacked = False
            for ff in os.listdir(huc_dir):
                if dt.split(".")[0].lower() in ff.lower():
                    unpacked = True
            if not unpacked:
                # Unpack the downloaded file
                unpack7z(filepath, huc_dir)
                print("Unpacked TAR ", fname, "\n")
            else:
                print("File should be already unpacked in %s" % huc_dir)


def mergeRasters(files, outfile):
    """ Function takes a given list of file paths and merges to given output file """
    print(files)
    print("Beginning merge of :\n\t%s" % ("\n\t".join(files)))

    with rio.open(files[0]) as ras:
        dtype = ras.read().dtype
        print(dtype)

    sources = [rio.open(f) for f in files]
    merged_array, output_transform = merge_tool(sources)

    profile = sources[0].profile
    profile['transform'] = output_transform
    profile['height'] = merged_array.shape[1]
    profile['width'] = merged_array.shape[2]

    profile.update(dtype=np.float32)
    #print(merged_array.dtype)
    #merged_array = merged_array.astype(dtype)
    #print(merged_array.dtype)

    # print(profile)

    print("Writing merged rasters out to %s outfile...")
    with rio.open(outfile, 'w', **profile) as dst:
        dst.write(merged_array.astype(np.float32))

    return outfile


def getFullNAIPPath(naip_file, naipdir):
    for root, dirs, files in os.walk(naipdir):
        for file in files:
            if naip_file in file:
                return os.path.join(root, file)

    print("Unable to find naip file %s in %s. Exiting" % (naip_file, naipdir))
    raise Exception


def getFilesonDE(base_path="/iplant/home/bhickson/2015/Data"):
    pw = "Cinco12#"
    session = iRODSSession(host='data.cyverse.org', zone="iplant", port=1247, user='bhickson', password=pw)

    data_col = session.collections.get(base_path)

    ifiles = {}
    def getFilesandDirs(dir):
        #print(dir.name)
        files_list = dir.data_objects
        dirs_list = dir.subcollections
        for file in files_list:
            file_name = file.name
            ifiles[file.name] = file.path
        for sub_dir in dirs_list:
            #print(sub_dir.name)
            getFilesandDirs(sub_dir)

    getFilesandDirs(data_col)

    return session, ifiles


def pushToDE(file_path, irods_files, session, base_datadir="../Data", irods_data_path="/iplant/home/bhickson/2015/Data"):
    # push comleted file to irods
    fname = os.path.basename(file_path)
    if fname not in irods_files.keys():
        relative_dir = file_path.split(os.path.abspath(base_datadir))[1]
        irods_odir = irods_data_path + relative_dir
        print("-------- Pushing data to DE....  -   %s" % irods_odir)
        session.data_objects.put(file_path, irods_odir)


def writeToGPKG(filename, df):
    """ Geopandas current implimentation is very slow to write data frame to file due to file, locking/unlock for
    each feature. Get around this with fionas buffer.

    This code pulled from https://github.com/geopandas/geopandas/issues/557"""

    g = df.columns.to_series().groupby(df.dtypes).groups
    properties = {}
    for k, v in g.items():
        for i in v:
            if i != 'geometry':
                # print(i)
                properties[i] = k.name

    file_schema = {'geometry': df.geom_type.tolist()[0],
                    'properties': {'Type': 'str',
                                   'Class': 'int',
                                   'PROJ': 'str',
                                   'NAIP_FILE': 'str'}}


    with fiona.drivers():
        with fiona.open(filename, 'w', driver="GPKG", crs=df.crs, schema=file_schema) as colxn:
            colxn.writerecords(df.iterfeatures())
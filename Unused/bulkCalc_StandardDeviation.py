# RASTER CALCULATION WHICH PRODUCES A FOCAL STANDARD DEVIATION RASTER. DEFAULTS TO 5 PIXEL WINDOWS DIAMETER

from datetime import datetime
import time
import os
from RasterCalculations import standardDeviation

naip_dir = r"Q:\Arid Riparian Project\Data\az_1m_2015"
out_loc = r"M:\Data"

start = time.process_time()

overwrite = False

# A counter for all files
fcount = 0

for root, dirs, files in os.walk(naip_dir):
    for file in files:
        if file.endswith(".tif"):
            fcount += 1
            print(fcount, datetime.now().strftime('%M:%S.%f')[:-4], "-", file)
            fpath = os.path.join(root, file)

            standardDeviation(fpath, out_loc, window_size=3, overwrite=False)
            standardDeviation(fpath, out_loc, window_size=5, overwrite=False)
            standardDeviation(fpath, out_loc, window_size=10, overwrite=False)

end = time.process_time()

print("Elapsed:", datetime.now() - start)
datetime.strftime()
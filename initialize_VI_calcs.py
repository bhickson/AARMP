import os
import time
from datetime import datetime

from RasterCalculations import vegIndexCalc

imagery_dir = r"Q:\Arid Riparian Project\Data\az_1m_2015"
out_loc = r"M:\Data"

start = time.process_time()

# A counter for all files
fcount = 0

for root, dirs, files in os.walk(imagery_dir):
    for file in files:
        if file.endswith(".tif"):
            fcount += 1
            print(fcount, datetime.now().strftime('%M:%S.%f')[:-4], "-", file)
            fpath = os.path.join(root, file)

            test_dir = os.path.join(out_loc, "NDVI")
            test_file = os.path.join(test_dir, "NDVI_" + file)
            if not os.path.exists(test_file):
                vegIndexCalc(fpath, out_loc, ["NDVI", "SAVI", "OSAVI", "MSAVI2", "EVI2"])

end = time.process_time()
print(end - start)



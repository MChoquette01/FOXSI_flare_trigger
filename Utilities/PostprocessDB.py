import numpy as np
from pymongo import MongoClient
import netCDF4 as nc
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import math
from multiprocessing import Pool


flare_filepath = r"C:\Users\matth\Documents\Capstone\data\GOES_XRS_historical.fits"
xray_filepath = r"C:\Users\matth\Documents\Capstone\data\sci_xrsf-l2-avg1m_g16_s20170207_e20240906_v2-2-0.nc"

def assign_flare_class(xrsb_history):

    max_flux = np.max(xrsb_history)
    flux_value_str = '%.12f' % float(max_flux)
    if max_flux < 10**-7:
        return f"A{flux_value_str[9]}.{flux_value_str[10]}"
    elif 10**-7 <= max_flux < 10**-6:
        return f"B{flux_value_str[8]}.{flux_value_str[9]}"
    elif 10**-6 <= max_flux < 10**-5:
        return f"C{flux_value_str[7]}.{flux_value_str[8]}"
    elif 10**-5 <= max_flux < 10**-4:
        return f"M{flux_value_str[6]}.{flux_value_str[7]}"
    elif max_flux > 10**-4:
        return f"X{flux_value_str[5]}.{flux_value_str[6]}"

# Open MongoDB
myclient = MongoClient("mongodb://localhost:27017/")
flares_db = myclient["Flares"]
flares_table = flares_db["FlaresUploadTest"]

# flares_table.update_one({'FlareID':"201702090041_0"}, {"$set": {"XRSA3MinuteDifference": None}})


# def fix_nans(datum):
#
#     nan_records = flares_table.find({datum: float(np.nan)})
#     for record in tqdm(nan_records, desc=f"Fixing {datum} NaNs"):
#         flares_table.update_one({'FlareID':record["FlareID"]}, {"$set": {datum: None}})
#
#
# if __name__ == "__main__":
#
#     datums = ["Temperature", "EmissionMeasure", "Temperature1MinuteDifference", "Temperature3MinuteDifference",
#               "Temperature5MinuteDifference", "EmissionMeasure1MinuteDifference", "EmissionMeasure3MinuteDifference",
#               "EmissionMeasure5MinuteDifference"]
#
#     all_records = flares_table.find({})
#     for record in tqdm(all_records):
#         for datum in datums:
#             if record[datum] is not None:
#                 if math.isnan(record[datum]):
#                     # foo = 2
#                     flares_table.update_one({'FlareID': record["FlareID"]}, {"$set": {datum: None}})
#
#     # with Pool(4) as p:
#     #     p.map(fix_nans, datums)
#     # for datum in datums:
#     #     fix_nans(datum)

# Open FITS file
flares = fits.open(flare_filepath)
data = Table(flares[1].data)[:]
all_data = []
for index in range(len(data)):
    this_data = {}
    for variable in data.colnames:
        this_data[variable] = data[variable][index]
    all_data.append(this_data)

no_flare_class_records = flares_table.find({"FlareClass": ""})
current_flare_id = None
for record in tqdm(no_flare_class_records, desc="Fixing empty flare classes"):
    flare_id = int(record["FlareID"].split("_")[0])
    if flare_id != current_flare_id:
        current_flare_id = flare_id
        fits_record = [x for x in all_data if x["flare ID"] == current_flare_id][0]
        flare_class = assign_flare_class(fits_record["xrsb"])
    flares_table.update_one({'FlareID':record["FlareID"]}, {"$set": {"FlareClass": flare_class}})

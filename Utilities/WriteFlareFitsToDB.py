from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import math
from pymongo import MongoClient
import emission_measure as emt
import re

"""Transfers data from a FITS file with extracted flares (from making_historical.py) to MongoDB.
Requires download of MongoDB (https://www.mongodb.com/try/download/community) and Mongo DB Compass (part of the default install)."""


flare_filepath = r"C:\Users\matth\Documents\Capstone\data\GOES_XRS_historical.fits"
xray_filepath = r"C:\Users\matth\Documents\Capstone\data\sci_xrsf-l2-avg1m_g16_s20170207_e20240906_v2-2-0.nc"
# flare_filepath = '../../GOES_XRS_historical.fits'
# xray_filepath = '../../sci_xrsf-l2-avg1m_g16_s20170207_e20240217_v2-2-0.nc'


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


print('trying to open xray file')
# open xray file
xray_data = nc.Dataset(xray_filepath)
xray_xrsa_fluxes = xray_data.variables["xrsa_flux_observed"][:]
xray_xrsa_fluxes = xray_xrsa_fluxes.tolist(fill_value=np.nan)
xray_xrsb_fluxes = xray_data.variables["xrsb_flux_observed"][:]
xray_xrsb_fluxes = xray_xrsb_fluxes.tolist(fill_value=np.nan)
xray_times = xray_data.variables["time"][:]
xray_times = xray_times.tolist(fill_value=np.nan)

# Open FITS file
flares = fits.open(flare_filepath)
data = Table(flares[1].data)[:]
all_data = []
for index in range(len(data)):
    this_data = {}
    for variable in data.colnames:
        this_data[variable] = data[variable][index]
    all_data.append(this_data)

# Open MongoDB
myclient = MongoClient("mongodb://localhost:27017/")
flares_db = myclient["Flares"]
flares_table = flares_db["Flares"]
flares_table.delete_many({})  # clear out whatever is in there

def calculate_differences(flare_data, xray_times, xray_xrsa_fluxes, xray_xrsb_fluxes, idx, n):
    ''' Calculates the n-minute differences for xrsa and xrsb
    '''
    if idx >= n:
        xrsa_n_minute_diff = float(flare_data["xrsa"][idx] - flare_data["xrsa"][idx - n])
        xrsb_n_minute_diff = float(flare_data["xrsb"][idx] - flare_data["xrsb"][idx - n])
    # ...unless we don't have that much data for a flare, then find the X-ray data from the source...
    else:
        xray_time_index = xray_times.index(time)
        xrsa_n_minute_diff = float(xray_xrsa_fluxes[xray_time_index] - xray_xrsa_fluxes[xray_time_index - n])
        xrsb_n_minute_diff = float(xray_xrsb_fluxes[xray_time_index] - xray_xrsb_fluxes[xray_time_index - n])
        # ...UNLESS it's still NaN (bad data)...write as null
        if math.isnan(xrsa_n_minute_diff):
            xrsa_n_minute_diff = None
        if math.isnan(xrsb_n_minute_diff):
            xrsb_n_minute_diff = None

    # check for funny errors
    if xrsa_n_minute_diff is not None and abs(xrsa_n_minute_diff) > 9000:
        xrsa_n_minute_diff = None
    if xrsb_n_minute_diff is not None and abs(xrsb_n_minute_diff) > 9000:
        xrsb_n_minute_diff = None
            
    return xrsa_n_minute_diff, xrsb_n_minute_diff

# Write FITS to DB
for flare_data in tqdm(all_data, desc="Writing Flares Database..."):
    flare_id = flare_data["flare ID"]
    flare_class = flare_data["class"]
    if flare_class == "":
        flare_class = assign_flare_class(flare_data["xrsb"])
    peak_time = flare_data["peak time"]
    for idx, (xrsa_datum, xrsb_datum, time) in enumerate(zip(flare_data["xrsa"], flare_data["xrsb"], flare_data["time"])):
        # TODO: Why are there -9999.0s?
        if xrsa_datum == -9999.0:
            xrsa_datum = None
        if xrsb_datum == -9999.0:
            xrsb_datum = None
        minutes_to_peak = (peak_time - flare_data["time"][idx]) / 60
        xrsa_remaining_data = str(flare_data["xrsa"][idx:])
        # convert data to be a list-like string
        # '[one two three]' -> '[one, two, three]'
        values = re.findall(r"\S{3,}", xrsa_remaining_data)
        values[0] = values[0].replace("[", "")
        values[-1] = values[-1].replace("]", "")
        xrsa_remaining_data = [float(x) for x in values]
        xrsb_remaining_data = str(flare_data["xrsb"][idx:])
        values = re.findall(r"\S{3,}", xrsb_remaining_data)
        values[0] = values[0].replace("[", "")
        values[-1] = values[-1].replace("]", "")
        xrsb_remaining_data = [float(x) for x in values]
        # calculate difference in quantities over the last 1,3,5 datapoints...
        xrsa_1_minute_diff, xrsb_1_minute_diff = calculate_differences(flare_data, xray_times, xray_xrsa_fluxes, xray_xrsb_fluxes, idx, 1)
        xrsa_3_minute_diff, xrsb_3_minute_diff = calculate_differences(flare_data, xray_times, xray_xrsa_fluxes, xray_xrsb_fluxes, idx, 3)
        xrsa_5_minute_diff, xrsb_5_minute_diff = calculate_differences(flare_data, xray_times, xray_xrsa_fluxes, xray_xrsb_fluxes, idx, 5)

        #add in the temperature and emission measure features:
        if xrsa_datum is not None and xrsb_datum is not None:
            emission_measure, temperature = emt.compute_goes_emission_measure(xrsa_datum, xrsb_datum, 16)
        else:
            emission_measure, temperature = [None], [None]
        if xrsa_datum is not None and xrsb_datum is not None:
            em_1_minute_diff, temp_1_minute_diff = emt.compute_goes_emission_measure(xrsa_1_minute_diff, xrsb_1_minute_diff, 16)
        else:
            em_1_minute_diff, temp_1_minute_diff = [None], [None]
        if xrsa_datum is not None and xrsb_datum is not None:
            em_3_minute_diff, temp_3_minute_diff = emt.compute_goes_emission_measure(xrsa_3_minute_diff, xrsb_3_minute_diff, 16)
        else:
            em_3_minute_diff, temp_3_minute_diff = [None], [None]
        if xrsa_datum is not None and xrsb_datum is not None:
            em_5_minute_diff, temp_5_minute_diff = emt.compute_goes_emission_measure(xrsa_5_minute_diff, xrsb_5_minute_diff, 16)
        else:
            em_5_minute_diff, temp_5_minute_diff = [None], [None]
        #editing format so that the non-null EM and Temp differences are inserted correctly.
        if em_1_minute_diff is not None:
            em_1_minute_diff = em_1_minute_diff[0]
            temp_1_minute_diff = temp_1_minute_diff[0]
        if em_3_minute_diff is not None:
            em_3_minute_diff = em_3_minute_diff[0]
            temp_3_minute_diff = temp_3_minute_diff[0]
        if em_5_minute_diff is not None:
            em_5_minute_diff = em_5_minute_diff[0]
            temp_5_minute_diff = temp_5_minute_diff[0]

        flares_table.insert_one({"_id": f"{flare_id}_{idx}",
                                 "FlareID": f"{flare_id}_{idx}",
                                 "FlareClass": flare_class,
                                 "MinutesToPeak": minutes_to_peak,
                                 "CurrentXRSA": float(xrsa_datum) if xrsa_datum is not None else None,
                                 "CurrentXRSB": float(xrsb_datum) if xrsb_datum is not None else None,
                                 "XRSA1MinuteDifference": xrsa_1_minute_diff,
                                 "XRSB1MinuteDifference": xrsb_1_minute_diff,
                                 "XRSA3MinuteDifference": xrsa_3_minute_diff,
                                 "XRSB3MinuteDifference": xrsb_3_minute_diff,
                                 "XRSA5MinuteDifference": xrsa_5_minute_diff,
                                 "XRSB5MinuteDifference": xrsb_5_minute_diff,
                                 "Temperature": temperature[0],
                                 "EmissionMeasure": emission_measure[0],
                                 "Temperature1MinuteDifference": temp_1_minute_diff,
                                 "EmissionMeasure1MinuteDifference": em_1_minute_diff,
                                 "Temperature3MinuteDifference": temp_3_minute_diff,
                                 "EmissionMeasure3MinuteDifference": em_3_minute_diff,
                                 "Temperature5MinuteDifference": temp_5_minute_diff,
                                 "EmissionMeasure5MinuteDifference": em_5_minute_diff,
                                 "XRSARemaining": xrsa_remaining_data,
                                 "XRSBRemaining": xrsb_remaining_data})

import pandas as pd
import numpy as np
import tree_common as tc
import netCDF4 as nc
from datetime import datetime
import sys
sys.path.insert(0, "../Utilities")
import emission_measure as em

# Creates lookup tables for temperature and emission measure 1, 3, and 5 minute differences.
# But unlike the scripts in Utilities/, it uses the difference between a flares temperature at time X and X - 5,
# preventing NaN values but losing some physical meaning.


xray_filepath = r"C:\Users\matth\Documents\Capstone\data\sci_xrsf-l2-avg1m_g16_s20170207_e20240906_v2-2-0.nc"


def get_diff(flare_id, timestamp, current_temp, current_em, difference_minutes):
    """Returns the temp and EM difference over <difference_minutes> minutes from teh current values"""

    if timestamp >= difference_minutes:  # can do everything with the DB
        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_{timestamp - difference_minutes}'}}, {"Temperature": 1,
                                                                                                              "EmissionMeasure": 1})
        for record in cursor:
            temp_diff = current_temp - record["Temperature"]
            em_diff = current_em - record["EmissionMeasure"]
    else:  # gonna need the xray file
        minutes_previous_em, minutes_previous_temp = em.compute_goes_emission_measure(xray_xrsa[xray_times_index - difference_minutes],
                                                                                      xray_xrsb[xray_times_index - difference_minutes],
                                                                                      goes_sat=16)
        minutes_previous_temp = minutes_previous_temp[0]
        minutes_previous_em = minutes_previous_em[0]
        temp_diff = current_temp - minutes_previous_temp
        em_diff = current_em - minutes_previous_em

    return temp_diff, em_diff


# read in xray data
xray_data = nc.Dataset(xray_filepath)
xray_times = xray_data.variables["time"][:].tolist()
xray_dates = nc.num2date(xray_times, units=xray_data.variables["time"].units)
xray_xrsa = xray_data.variables["xrsa_flux"][:].tolist()
xray_xrsb = xray_data.variables["xrsb_flux"][:].tolist()

client, flares_table = tc.connect_to_flares_db()

diffs = []  # store results
# get unique flare IDs
flare_ids = [x["FlareID"].split("_")[0] for x in flares_table.find({'FlareID': {'$regex': '_0$'}})]
for flare_id in flare_ids:
    this_flare_diffs = []
    flare_records = flares_table.count_documents({'FlareID': {'$regex': f'^{flare_id}'}})
    for timestamp in range(flare_records):
        # get the temp and EM at this timestamp for this flare
        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_{timestamp}'}}, {"Temperature": 1,
                                                                                         "EmissionMeasure": 1})
        for record in cursor:
            current_temp = record["Temperature"]
            current_em = record["EmissionMeasure"]

        # if timestamp less than 5, we'll need to read the Xray file at some point since that info isn't in the DB
        if timestamp <= 5:
            # parse flare ID into the date it is
            year = flare_id[:4]
            month = flare_id[4:6]
            day = flare_id[6:8]
            hour = flare_id[8:10]
            minute = flare_id[10:12]
            # now into a datetime
            flare_datetime = datetime.strptime(f"{year}-{month}-{day}-{hour}-{minute}", "%Y-%m-%d-%H-%M")
            xray_dates = nc.date2num([flare_datetime], units=xray_data.variables["time"].units)
            # find it's index in the xray file
            xray_times_index = xray_times.index(float(xray_dates[0]))

        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_{timestamp - 1}'}}, {"Temperature": 1,
                                                                                             "EmissionMeasure": 1})

        one_minute_temp_diff, one_minute_em_diff = get_diff(flare_id, timestamp, current_temp, current_em, difference_minutes=1)
        three_minute_temp_diff, three_minute_em_diff = get_diff(flare_id, timestamp, current_temp, current_em, difference_minutes=3)
        five_minute_temp_diff, five_minute_em_diff = get_diff(flare_id, timestamp, current_temp, current_em, difference_minutes=5)

        this_flare_diffs = [flare_id, timestamp, one_minute_temp_diff, one_minute_em_diff, three_minute_temp_diff,
                            three_minute_em_diff, five_minute_temp_diff, five_minute_em_diff]
        diffs.append(this_flare_diffs)

results = pd.DataFrame(np.array(diffs))
results.columns = ["Flare ID", "Timestamp", "Temperature1MinuteDifference", "EmissionMeasure1MinuteDifference",
                   "Temperature3MinuteDifference", "EmissionMeasure3MinuteDifference",
                   "Temperature5MinuteDifference", "EmissionMeasure5MinuteDifference"]
client.close()

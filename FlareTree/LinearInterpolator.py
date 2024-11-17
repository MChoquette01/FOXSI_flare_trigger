import tree_common as tc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import sys
import argparse

"""Create DataFrames that act as lookup tables for a given timestamp/flare when interpolating linearly to fill NaNs"""


def pad_list(datum_list):
    """Takes a list of lists and appends NaNs to the shorter lists such that all are the same length."""

    max_length = max([len(x) for x in datum_list])
    for idx in range(len(datum_list)):
        datum_list[idx] += [np.nan] * (max_length - len(datum_list[idx]))

    return datum_list


def do_interpolate(to_interpolate, flare_ids, variable_name, minutes_since_flare_start):

    x = pad_list(to_interpolate)
    x = pd.DataFrame(np.array(x), dtype=np.float64).T
    x = x.interpolate(method='linear', limit_direction='both')
    x.columns = flare_ids

    if not os.path.exists("Interpolations"):
        os.mkdir("Interpolations")

    if not os.path.exists(os.path.join("Interpolations", f"{minutes_since_flare_start}_minutes_since_start")):
        os.mkdir(os.path.join("Interpolations", f"{minutes_since_flare_start}_minutes_since_start"))

    with open(os.path.join("Interpolations", f"{minutes_since_flare_start}_minutes_since_start", f"{variable_name}.pkl"), "wb") as f:
        pickle.dump(x, f)


def create_interpolated_table_for_timestamp(minutes_since_flare_start):

    uninterpolated_values = defaultdict(list)
    client, flares_table = tc.connect_to_flares_db()
    # get unique flare IDs
    flare_ids = [x["FlareID"].split("_")[0] for x in flares_table.find({'FlareID': {'$regex': f'_{minutes_since_flare_start + 15}$'}})]
    for flare_id in tqdm(flare_ids):
        this_record = defaultdict(list)
        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_'}}, {"FlareID": 1,
                                                                              "Temperature": 1,
                                                                              "Temperature1MinuteDifference": 1,
                                                                              "Temperature3MinuteDifference": 1,
                                                                              "Temperature5MinuteDifference": 1,
                                                                              "EmissionMeasure": 1,
                                                                              "EmissionMeasure1MinuteDifference": 1,
                                                                              "EmissionMeasure3MinuteDifference": 1,
                                                                              "EmissionMeasure5MinuteDifference": 1,
                                                                              "CurrentXRSA": 1,
                                                                              "CurrentXRSB": 1})

        for flare_entry in cursor:
            # make a list of all records from this flare
            # cursor is a generator, so loop through each record and extract all values
            this_record["Temperature"].append(flare_entry["Temperature"])
            this_record["Temperature1MinuteDifference"].append(flare_entry["Temperature1MinuteDifference"])
            this_record["Temperature3MinuteDifference"].append(flare_entry["Temperature3MinuteDifference"])
            this_record["Temperature5MinuteDifference"].append(flare_entry["Temperature5MinuteDifference"])
            this_record["EmissionMeasure"].append(flare_entry["EmissionMeasure"])
            this_record["EmissionMeasure1MinuteDifference"].append(flare_entry["EmissionMeasure1MinuteDifference"])
            this_record["EmissionMeasure3MinuteDifference"].append(flare_entry["EmissionMeasure3MinuteDifference"])
            this_record["EmissionMeasure5MinuteDifference"].append(flare_entry["EmissionMeasure5MinuteDifference"])
            this_record["CurrentXRSA"].append(flare_entry["CurrentXRSA"])
            this_record["CurrentXRSB"].append(flare_entry["CurrentXRSB"])
        # add this flare's values to the running list
        uninterpolated_values["Temperature"].append(this_record["Temperature"])
        uninterpolated_values["Temperature1MinuteDifference"].append(this_record["Temperature1MinuteDifference"])
        uninterpolated_values["Temperature3MinuteDifference"].append(this_record["Temperature3MinuteDifference"])
        uninterpolated_values["Temperature5MinuteDifference"].append(this_record["Temperature5MinuteDifference"])
        uninterpolated_values["EmissionMeasure"].append(this_record["EmissionMeasure"])
        uninterpolated_values["EmissionMeasure1MinuteDifference"].append(this_record["EmissionMeasure1MinuteDifference"])
        uninterpolated_values["EmissionMeasure3MinuteDifference"].append(this_record["EmissionMeasure3MinuteDifference"])
        uninterpolated_values["EmissionMeasure5MinuteDifference"].append(this_record["EmissionMeasure5MinuteDifference"])
        uninterpolated_values["CurrentXRSA"].append(this_record["CurrentXRSA"])
        uninterpolated_values["CurrentXRSB"].append(this_record["CurrentXRSB"])
    client.close()

    # do the actual interpolating
    do_interpolate(uninterpolated_values["Temperature"], flare_ids, "Temperature", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["Temperature1MinuteDifference"], flare_ids, "Temperature1MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["Temperature3MinuteDifference"], flare_ids, "Temperature3MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["Temperature5MinuteDifference"], flare_ids, "Temperature5MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["EmissionMeasure"], flare_ids, "EmissionMeasure", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["EmissionMeasure1MinuteDifference"], flare_ids, "EmissionMeasure1MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["EmissionMeasure3MinuteDifference"], flare_ids, "EmissionMeasure3MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["EmissionMeasure5MinuteDifference"], flare_ids, "EmissionMeasure5MinuteDifference", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["CurrentXRSA"], flare_ids, "CurrentXRSA", minutes_since_flare_start)
    do_interpolate(uninterpolated_values["CurrentXRSB"], flare_ids, "CurrentXRSB", minutes_since_flare_start)


if __name__ == "__main__":

    # running from MSI/Slurm, or CMD line, if you really want to
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", type=int, help="Timestamp relative to start of flare to interpolate for")
        args = parser.parse_args()
        create_interpolated_table_for_timestamp(minutes_since_flare_start=args.t)
    else:
        for timestamp in range(-5, 4, 1):
            create_interpolated_table_for_timestamp(minutes_since_flare_start=timestamp)
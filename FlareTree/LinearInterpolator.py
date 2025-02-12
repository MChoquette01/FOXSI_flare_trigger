import tree_common as tc
import numpy as np
import pandas as pd
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


def do_interpolate(to_interpolate, flare_ids, variable_name):

    x = pad_list(to_interpolate)
    x = pd.DataFrame(np.array(x), dtype=np.float64).T
    x = x.interpolate(method='linear', limit_direction='both')
    x.columns = flare_ids

    if not os.path.exists("Interpolations"):
        os.mkdir("Interpolations")

    with open(os.path.join("Interpolations", f"{variable_name}.pkl"), "wb") as f:
        pickle.dump(x, f)


def create_interpolated_table_for_timestamp(use_naive_diffs=False):

    uninterpolated_values = defaultdict(list)
    client, flares_table = tc.connect_to_flares_db(use_naive=use_naive_diffs)
    # get unique flare IDs
    flare_ids = [x["FlareID"].split("_")[0] for x in flares_table.find({'FlareID': {'$regex': '_0$'}})]
    for flare_id in flare_ids:
        if int(flare_id) in tc.BLACKLISTED_FLARE_IDS:
            continue
        this_record = defaultdict(list)
        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_'}}, {"XRSARemaining": 0,
                                                                              "XRSBRemaining": 0})

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
            this_record["XRSA1MinuteDifference"].append(flare_entry["XRSA1MinuteDifference"])
            this_record["XRSA3MinuteDifference"].append(flare_entry["XRSA3MinuteDifference"])
            this_record["XRSA5MinuteDifference"].append(flare_entry["XRSA5MinuteDifference"])
            this_record["XRSB1MinuteDifference"].append(flare_entry["XRSB1MinuteDifference"])
            this_record["XRSB3MinuteDifference"].append(flare_entry["XRSB3MinuteDifference"])
            this_record["XRSB5MinuteDifference"].append(flare_entry["XRSB5MinuteDifference"])
            if use_naive_diffs:
                this_record["NaiveTemperature1MinuteDifference"].append(flare_entry["NaiveTemperature1MinuteDifference"])
                this_record["NaiveTemperature3MinuteDifference"].append(flare_entry["NaiveTemperature3MinuteDifference"])
                this_record["NaiveTemperature5MinuteDifference"].append(flare_entry["NaiveTemperature5MinuteDifference"])
                this_record["NaiveEmissionMeasure1MinuteDifference"].append(flare_entry["NaiveEmissionMeasure1MinuteDifference"])
                this_record["NaiveEmissionMeasure3MinuteDifference"].append(flare_entry["NaiveEmissionMeasure3MinuteDifference"])
                this_record["NaiveEmissionMeasure5MinuteDifference"].append(flare_entry["NaiveEmissionMeasure5MinuteDifference"])
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
        uninterpolated_values["XRSA1MinuteDifference"].append(this_record["XRSA1MinuteDifference"])
        uninterpolated_values["XRSA3MinuteDifference"].append(this_record["XRSA3MinuteDifference"])
        uninterpolated_values["XRSA5MinuteDifference"].append(this_record["XRSA5MinuteDifference"])
        uninterpolated_values["XRSB1MinuteDifference"].append(this_record["XRSB1MinuteDifference"])
        uninterpolated_values["XRSB3MinuteDifference"].append(this_record["XRSB3MinuteDifference"])
        uninterpolated_values["XRSB5MinuteDifference"].append(this_record["XRSB5MinuteDifference"])
        if use_naive_diffs:
            uninterpolated_values["NaiveTemperature1MinuteDifference"].append(this_record["NaiveTemperature1MinuteDifference"])
            uninterpolated_values["NaiveTemperature3MinuteDifference"].append(this_record["NaiveTemperature3MinuteDifference"])
            uninterpolated_values["NaiveTemperature5MinuteDifference"].append(this_record["NaiveTemperature5MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure1MinuteDifference"].append(this_record["NaiveEmissionMeasure1MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure3MinuteDifference"].append(this_record["NaiveEmissionMeasure3MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure5MinuteDifference"].append(this_record["NaiveEmissionMeasure5MinuteDifference"])
    client.close()

    # do the actual interpolating
    do_interpolate(uninterpolated_values["Temperature"], flare_ids, "Temperature")
    do_interpolate(uninterpolated_values["Temperature1MinuteDifference"], flare_ids, "Temperature1MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature3MinuteDifference"], flare_ids, "Temperature3MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature5MinuteDifference"], flare_ids, "Temperature5MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure"], flare_ids, "EmissionMeasure")
    do_interpolate(uninterpolated_values["EmissionMeasure1MinuteDifference"], flare_ids, "EmissionMeasure1MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure3MinuteDifference"], flare_ids, "EmissionMeasure3MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure5MinuteDifference"], flare_ids, "EmissionMeasure5MinuteDifference")
    do_interpolate(uninterpolated_values["CurrentXRSA"], flare_ids, "CurrentXRSA")
    do_interpolate(uninterpolated_values["CurrentXRSB"], flare_ids, "CurrentXRSB")
    do_interpolate(uninterpolated_values["XRSA1MinuteDifference"], flare_ids, "XRSA1MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA3MinuteDifference"], flare_ids, "XRSA3MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA5MinuteDifference"], flare_ids, "XRSA5MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB1MinuteDifference"], flare_ids, "XRSB1MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB3MinuteDifference"], flare_ids, "XRSB3MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB5MinuteDifference"], flare_ids, "XRSB5MinuteDifference")
    if use_naive_diffs:
        do_interpolate(uninterpolated_values["NaiveTemperature1MinuteDifference"], flare_ids, "NaiveTemperature1MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature3MinuteDifference"], flare_ids, "NaiveTemperature3MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature5MinuteDifference"], flare_ids, "NaiveTemperature5MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure1MinuteDifference"], flare_ids, "NaiveEmissionMeasure1MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure3MinuteDifference"], flare_ids, "NaiveEmissionMeasure3MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure5MinuteDifference"], flare_ids, "NaiveEmissionMeasure5MinuteDifference")


if __name__ == "__main__":

    use_naive_diffs = True
    create_interpolated_table_for_timestamp(use_naive_diffs)
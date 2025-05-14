import tree_common as tc
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import pickle
import sys
import argparse
from copy import deepcopy

"""Create DataFrames that act as lookup tables for a given timestamp/flare when interpolating linearly to fill NaNs"""


def pad_list(datum_list):
    """Takes a list of lists and appends NaNs to the shorter lists such that all are the same length."""

    max_length = max([len(x) for x in datum_list])
    for idx in range(len(datum_list)):
        datum_list[idx] += [np.nan] * (max_length - len(datum_list[idx]))

    return datum_list


def do_interpolate(to_interpolate, flare_ids, variable_name):
    """Conduct linear interpolation on a set of flare variables"""

    x = pad_list(to_interpolate)
    x = pd.DataFrame(np.array(x), dtype=np.float64).T
    for time_minutes in range(10, 31):
        timestamp_flare_ids = deepcopy(flare_ids)
        x_truncated = x.iloc[:time_minutes + 1, :]  # only look at information known at this point in time, like live use!
        x_truncated = x_truncated.interpolate(method='linear', limit_direction='both')
        x_truncated.columns = timestamp_flare_ids
        # remove blacklisted flare IDs from column labels - flares with all NaNs in at least one column
        blacklist_pkl_filepath = rf"Interpolations\BlacklistedFlares\{time_minutes - 15}_minutes_since_start.pkl"
        if os.path.exists(blacklist_pkl_filepath):
            with open(blacklist_pkl_filepath, "rb") as f:
                bad_flares_for_timestamp = pickle.load(f)
            for bad_flare_id in bad_flares_for_timestamp:
                if str(bad_flare_id) in timestamp_flare_ids:
                    timestamp_flare_ids.remove(str(bad_flare_id))
                    x_truncated = x_truncated.drop(columns=[bad_flare_id])

        out_dir = os.path.join("Interpolations", f"{time_minutes - 15}_minutes_since_start")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, f"{variable_name}.pkl"), "wb") as f:
            pickle.dump(x_truncated, f)


def create_interpolated_table_for_timestamp(use_naive_diffs=False):
    """Extract data from database to interpolate over"""

    uninterpolated_values = defaultdict(list)
    client, flares_table = tc.connect_to_flares_db(use_naive=use_naive_diffs)
    # get unique flare IDs
    flare_ids = [x["FlareID"].split("_")[0] for x in flares_table.find({'FlareID': {'$regex': '_0$'}})]
    for flare_id in flare_ids:
        this_record = defaultdict(list)
        cursor = flares_table.find({'FlareID': {'$regex': f'^{flare_id}_'}}, {"XRSARemaining": 0,
                                                                              "XRSBRemaining": 0})

        for flare_entry in cursor:
            # make a list of all records from this flare
            # cursor is a generator, so loop through each record and extract all values
            this_record["Temperature"].append(flare_entry["Temperature"])
            this_record["Temperature1MinuteDifference"].append(flare_entry["Temperature1MinuteDifference"])
            this_record["Temperature2MinuteDifference"].append(flare_entry["Temperature2MinuteDifference"])
            this_record["Temperature3MinuteDifference"].append(flare_entry["Temperature3MinuteDifference"])
            this_record["Temperature4MinuteDifference"].append(flare_entry["Temperature4MinuteDifference"])
            this_record["Temperature5MinuteDifference"].append(flare_entry["Temperature5MinuteDifference"])
            this_record["EmissionMeasure"].append(flare_entry["EmissionMeasure"])
            this_record["EmissionMeasure1MinuteDifference"].append(flare_entry["EmissionMeasure1MinuteDifference"])
            this_record["EmissionMeasure2MinuteDifference"].append(flare_entry["EmissionMeasure2MinuteDifference"])
            this_record["EmissionMeasure3MinuteDifference"].append(flare_entry["EmissionMeasure3MinuteDifference"])
            this_record["EmissionMeasure4MinuteDifference"].append(flare_entry["EmissionMeasure4MinuteDifference"])
            this_record["EmissionMeasure5MinuteDifference"].append(flare_entry["EmissionMeasure5MinuteDifference"])
            this_record["CurrentXRSA"].append(flare_entry["CurrentXRSA"])
            this_record["CurrentXRSB"].append(flare_entry["CurrentXRSB"])
            this_record["XRSBBackgroundFluxDifference"].append(flare_entry["XRSBBackgroundFluxDifference"])
            this_record["XRSA1MinuteDifference"].append(flare_entry["XRSA1MinuteDifference"])
            this_record["XRSA2MinuteDifference"].append(flare_entry["XRSA2MinuteDifference"])
            this_record["XRSA3MinuteDifference"].append(flare_entry["XRSA3MinuteDifference"])
            this_record["XRSA4MinuteDifference"].append(flare_entry["XRSA4MinuteDifference"])
            this_record["XRSA5MinuteDifference"].append(flare_entry["XRSA5MinuteDifference"])
            this_record["XRSB1MinuteDifference"].append(flare_entry["XRSB1MinuteDifference"])
            this_record["XRSB2MinuteDifference"].append(flare_entry["XRSB2MinuteDifference"])
            this_record["XRSB3MinuteDifference"].append(flare_entry["XRSB3MinuteDifference"])
            this_record["XRSB4MinuteDifference"].append(flare_entry["XRSB4MinuteDifference"])
            this_record["XRSB5MinuteDifference"].append(flare_entry["XRSB5MinuteDifference"])
            if use_naive_diffs:
                this_record["NaiveTemperature1MinuteDifference"].append(flare_entry["NaiveTemperature1MinuteDifference"])
                this_record["NaiveTemperature2MinuteDifference"].append(flare_entry["NaiveTemperature2MinuteDifference"])
                this_record["NaiveTemperature3MinuteDifference"].append(flare_entry["NaiveTemperature3MinuteDifference"])
                this_record["NaiveTemperature4MinuteDifference"].append(flare_entry["NaiveTemperature4MinuteDifference"])
                this_record["NaiveTemperature5MinuteDifference"].append(flare_entry["NaiveTemperature5MinuteDifference"])
                this_record["NaiveEmissionMeasure1MinuteDifference"].append(flare_entry["NaiveEmissionMeasure1MinuteDifference"])
                this_record["NaiveEmissionMeasure2MinuteDifference"].append(flare_entry["NaiveEmissionMeasure2MinuteDifference"])
                this_record["NaiveEmissionMeasure3MinuteDifference"].append(flare_entry["NaiveEmissionMeasure3MinuteDifference"])
                this_record["NaiveEmissionMeasure4MinuteDifference"].append(flare_entry["NaiveEmissionMeasure4MinuteDifference"])
                this_record["NaiveEmissionMeasure5MinuteDifference"].append(flare_entry["NaiveEmissionMeasure5MinuteDifference"])
        # add this flare's values to the running list
        uninterpolated_values["Temperature"].append(this_record["Temperature"])
        uninterpolated_values["Temperature1MinuteDifference"].append(this_record["Temperature1MinuteDifference"])
        uninterpolated_values["Temperature2MinuteDifference"].append(this_record["Temperature2MinuteDifference"])
        uninterpolated_values["Temperature3MinuteDifference"].append(this_record["Temperature3MinuteDifference"])
        uninterpolated_values["Temperature4MinuteDifference"].append(this_record["Temperature4MinuteDifference"])
        uninterpolated_values["Temperature5MinuteDifference"].append(this_record["Temperature5MinuteDifference"])
        uninterpolated_values["EmissionMeasure"].append(this_record["EmissionMeasure"])
        uninterpolated_values["EmissionMeasure1MinuteDifference"].append(this_record["EmissionMeasure1MinuteDifference"])
        uninterpolated_values["EmissionMeasure2MinuteDifference"].append(this_record["EmissionMeasure2MinuteDifference"])
        uninterpolated_values["EmissionMeasure3MinuteDifference"].append(this_record["EmissionMeasure3MinuteDifference"])
        uninterpolated_values["EmissionMeasure4MinuteDifference"].append(this_record["EmissionMeasure4MinuteDifference"])
        uninterpolated_values["EmissionMeasure5MinuteDifference"].append(this_record["EmissionMeasure5MinuteDifference"])
        uninterpolated_values["CurrentXRSA"].append(this_record["CurrentXRSA"])
        uninterpolated_values["CurrentXRSB"].append(this_record["CurrentXRSB"])
        uninterpolated_values["XRSBBackgroundFluxDifference"].append(this_record["XRSBBackgroundFluxDifference"])
        uninterpolated_values["XRSA1MinuteDifference"].append(this_record["XRSA1MinuteDifference"])
        uninterpolated_values["XRSA2MinuteDifference"].append(this_record["XRSA2MinuteDifference"])
        uninterpolated_values["XRSA3MinuteDifference"].append(this_record["XRSA3MinuteDifference"])
        uninterpolated_values["XRSA4MinuteDifference"].append(this_record["XRSA4MinuteDifference"])
        uninterpolated_values["XRSA5MinuteDifference"].append(this_record["XRSA5MinuteDifference"])
        uninterpolated_values["XRSB1MinuteDifference"].append(this_record["XRSB1MinuteDifference"])
        uninterpolated_values["XRSB2MinuteDifference"].append(this_record["XRSB2MinuteDifference"])
        uninterpolated_values["XRSB3MinuteDifference"].append(this_record["XRSB3MinuteDifference"])
        uninterpolated_values["XRSB4MinuteDifference"].append(this_record["XRSB4MinuteDifference"])
        uninterpolated_values["XRSB5MinuteDifference"].append(this_record["XRSB5MinuteDifference"])
        if use_naive_diffs:
            uninterpolated_values["NaiveTemperature1MinuteDifference"].append(this_record["NaiveTemperature1MinuteDifference"])
            uninterpolated_values["NaiveTemperature2MinuteDifference"].append(this_record["NaiveTemperature2MinuteDifference"])
            uninterpolated_values["NaiveTemperature3MinuteDifference"].append(this_record["NaiveTemperature3MinuteDifference"])
            uninterpolated_values["NaiveTemperature4MinuteDifference"].append(this_record["NaiveTemperature4MinuteDifference"])
            uninterpolated_values["NaiveTemperature5MinuteDifference"].append(this_record["NaiveTemperature5MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure1MinuteDifference"].append(this_record["NaiveEmissionMeasure1MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure2MinuteDifference"].append(this_record["NaiveEmissionMeasure2MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure3MinuteDifference"].append(this_record["NaiveEmissionMeasure3MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure4MinuteDifference"].append(this_record["NaiveEmissionMeasure4MinuteDifference"])
            uninterpolated_values["NaiveEmissionMeasure5MinuteDifference"].append(this_record["NaiveEmissionMeasure5MinuteDifference"])
    client.close()

    # do the actual interpolating
    do_interpolate(uninterpolated_values["Temperature"], flare_ids, "Temperature")
    do_interpolate(uninterpolated_values["Temperature1MinuteDifference"], flare_ids, "Temperature1MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature2MinuteDifference"], flare_ids, "Temperature2MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature3MinuteDifference"], flare_ids, "Temperature3MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature4MinuteDifference"], flare_ids, "Temperature4MinuteDifference")
    do_interpolate(uninterpolated_values["Temperature5MinuteDifference"], flare_ids, "Temperature5MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure"], flare_ids, "EmissionMeasure")
    do_interpolate(uninterpolated_values["EmissionMeasure1MinuteDifference"], flare_ids, "EmissionMeasure1MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure2MinuteDifference"], flare_ids, "EmissionMeasure2MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure3MinuteDifference"], flare_ids, "EmissionMeasure3MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure4MinuteDifference"], flare_ids, "EmissionMeasure4MinuteDifference")
    do_interpolate(uninterpolated_values["EmissionMeasure5MinuteDifference"], flare_ids, "EmissionMeasure5MinuteDifference")
    do_interpolate(uninterpolated_values["CurrentXRSA"], flare_ids, "CurrentXRSA")
    do_interpolate(uninterpolated_values["CurrentXRSB"], flare_ids, "CurrentXRSB")
    do_interpolate(uninterpolated_values["XRSBBackgroundFluxDifference"], flare_ids, "XRSBBackgroundFluxDifference")
    do_interpolate(uninterpolated_values["XRSA1MinuteDifference"], flare_ids, "XRSA1MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA2MinuteDifference"], flare_ids, "XRSA2MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA3MinuteDifference"], flare_ids, "XRSA3MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA4MinuteDifference"], flare_ids, "XRSA4MinuteDifference")
    do_interpolate(uninterpolated_values["XRSA5MinuteDifference"], flare_ids, "XRSA5MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB1MinuteDifference"], flare_ids, "XRSB1MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB2MinuteDifference"], flare_ids, "XRSB2MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB3MinuteDifference"], flare_ids, "XRSB3MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB4MinuteDifference"], flare_ids, "XRSB4MinuteDifference")
    do_interpolate(uninterpolated_values["XRSB5MinuteDifference"], flare_ids, "XRSB5MinuteDifference")
    if use_naive_diffs:
        do_interpolate(uninterpolated_values["NaiveTemperature1MinuteDifference"], flare_ids, "NaiveTemperature1MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature2MinuteDifference"], flare_ids, "NaiveTemperature2MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature3MinuteDifference"], flare_ids, "NaiveTemperature3MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature4MinuteDifference"], flare_ids, "NaiveTemperature4MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveTemperature5MinuteDifference"], flare_ids, "NaiveTemperature5MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure1MinuteDifference"], flare_ids, "NaiveEmissionMeasure1MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure2MinuteDifference"], flare_ids, "NaiveEmissionMeasure2MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure3MinuteDifference"], flare_ids, "NaiveEmissionMeasure3MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure4MinuteDifference"], flare_ids, "NaiveEmissionMeasure4MinuteDifference")
        do_interpolate(uninterpolated_values["NaiveEmissionMeasure5MinuteDifference"], flare_ids, "NaiveEmissionMeasure5MinuteDifference")


cols = ["CurrentXRSA", "XRSBBackgroundFluxDifference", "XRSA1MinuteDifference",
           "XRSA2MinuteDifference", "XRSA3MinuteDifference", "XRSA4MinuteDifference", "XRSA5MinuteDifference", "CurrentXRSB",
           "XRSB1MinuteDifference", "XRSB2MinuteDifference", "XRSB3MinuteDifference", "XRSB4MinuteDifference", "XRSB5MinuteDifference",
           "Temperature", "Temperature1MinuteDifference", "Temperature2MinuteDifference", "Temperature3MinuteDifference", "Temperature4MinuteDifference",
           "Temperature5MinuteDifference", "EmissionMeasure", "EmissionMeasure1MinuteDifference", "EmissionMeasure2MinuteDifference",
           "EmissionMeasure3MinuteDifference", "EmissionMeasure4MinuteDifference", "EmissionMeasure5MinuteDifference",
           "NaiveTemperature1MinuteDifference", "NaiveTemperature2MinuteDifference", "NaiveTemperature3MinuteDifference", "NaiveTemperature4MinuteDifference",
           "NaiveTemperature5MinuteDifference", "NaiveEmissionMeasure1MinuteDifference", "NaiveEmissionMeasure2MinuteDifference",
           "NaiveEmissionMeasure3MinuteDifference", "NaiveEmissionMeasure4MinuteDifference", "NaiveEmissionMeasure5MinuteDifference"]


def check_for_bad_flares():
    """Find flares for which all values are NaN in at least one column.
    These are not able to be used in Random Forests and Gradient Boosted Trees"""
    for time_minutes in range(-5, 16):
        bad_flares_for_timestamp = []
        for col in cols:
            pkl_filepath = fr"C:Interpolations\{time_minutes}_minutes_since_start\{col}.pkl"
            with open(pkl_filepath, "rb") as f:
                pkl = pickle.load(f)
            nan_cols = pkl.columns[pkl.isna().all()].tolist()  # columns here are Flare IDs
            if nan_cols:
                for x in nan_cols:
                    bad_flares_for_timestamp.append(x)

        out_path = "Interpolations/BlacklistedFlares"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        bad_flares_for_timestamp = list(set(bad_flares_for_timestamp))

        with open(os.path.join(out_path, rf"{time_minutes}_minutes_since_start.pkl"), "wb") as f:
            pickle.dump(bad_flares_for_timestamp, f)


if __name__ == "__main__":

    use_naive_diffs = True
    create_interpolated_table_for_timestamp(use_naive_diffs)
    check_for_bad_flares()
    create_interpolated_table_for_timestamp(use_naive_diffs)

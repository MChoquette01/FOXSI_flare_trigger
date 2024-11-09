import numpy as np
import pandas as pd
import pickle
from pymongo import MongoClient
import FlarePredictor as p

"""Run multiple threshold types with different threshold values.
Outputs a Pandas DataFarme for each threshold type with all flares/params tests"""


def batch_runner(threshold_class, xrsb_range, three_minute_difference_range):

    flare_ids = p.get_unique_flare_ids(flares_table)

    result_array = np.empty(shape=(0, 13), dtype=float)
    for xrsb_bin_size in xrsb_range:
        for difference_bin_size in three_minute_difference_range:
            t = threshold_class(xrsb_flux_step=xrsb_bin_size, xrsb_difference_step=difference_bin_size)
            for flare_index in p.get_unique_flare_count(flares_table):
                flare_id = flare_ids[flare_index]
                for flare_timestamp in range(0, p.get_timestamp_count_for_flare(flares_table, flare_id)):
                    base_flare = p.find_base_flare(flares_table,
                                                   flare_ids,
                                                   flare_index,
                                                   flare_timestamp)
                    t.calculate_thresholds(base_flare["xrsb_flux"],
                                           base_flare["xrsb_three_minute_difference"])
                    similar_flares = p.find_similar_flares(base_flare["flare_id"], flares_table, t.thresholds)
                    similar_flare_classes, class_text = p.analyze_similar_flare_classes(similar_flares)
                    result_array = np.append(result_array, [[xrsb_bin_size,
                                                             difference_bin_size,
                                                             base_flare["flare_id"],
                                                             flare_timestamp,
                                                             base_flare["flare_class"],
                                                             base_flare["peak_timestamp"] - flare_timestamp,
                                                             similar_flare_classes["A"],
                                                             similar_flare_classes["B"],
                                                             similar_flare_classes["C < 5"],
                                                             similar_flare_classes["C >= 5"],
                                                             similar_flare_classes["M"],
                                                             similar_flare_classes["X"],
                                                             similar_flare_classes["Unknown"]]], axis=0)
    result_df = pd.DataFrame(result_array)
    result_df.columns = ["XRSBFluxBinSize", "XRSB3MinuteDifferenceBinSize", "FlareID", "FlareTimestamp",
                         "BaseFlareClass", "MinutesToPeak", "ACount", "BCount", "C<5Count", "C>=5Count", "MCount", "XCount",
                         "UnknownClass"]
    with open(f"{t.threshold_type}_df.pkl", "wb") as f:
        pickle.dump(result_df, f)


if __name__ == "__main__":

    linear_xrsb_range = np.arange(10**-9, 10**-7, step=10**-9)
    linear_three_minute_difference_range = np.arange(10**-9, 10**-7, step=10**-9)
    log_xrsb_range = np.arange(10**-9, 10**-7, step=10**-9)
    log_three_minute_difference_range = np.arange(10 ** -9, 10 ** -7, step=10 ** -9)
    percent_xrsb_range = np.arange(0.1, 10, step=0.1)
    percent_three_minute_difference_range = np.arange(0.1, 10, step=0.1)

    client = MongoClient("mongodb://localhost:27017/")
    flares_db = client["Flares"]
    flares_table = flares_db["Flares"]

    batch_runner(p.LinearBinThreshold, linear_xrsb_range, linear_three_minute_difference_range)
    batch_runner(p.LinearMedianThreshold, linear_xrsb_range, linear_three_minute_difference_range)
    batch_runner(p.LogThreshold, log_xrsb_range, log_three_minute_difference_range)
    batch_runner(p.PercentThreshold, percent_xrsb_range, percent_three_minute_difference_range)

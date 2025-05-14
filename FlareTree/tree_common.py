from pymongo import MongoClient
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle


"""Common utilities for decision trees"""

RANDOM_STATE = 102024  # just the date I wrote this :)
LAUNCH_TIME_MINUTES = 8  # 3 minute XRS data latency + 3 minute countdown + 2 minute travel time
OBSERVATION_TIME_MINUTES = 6


def connect_to_flares_db(use_naive=False):
    """Connect to local MongoDB Flares DB"""

    client = MongoClient("mongodb://localhost:27017/")
    flares_db = client["Flares"]
    flares_table = flares_db["NaiveFlares"]

    return client, flares_table


def get_temporal_test_set_flare_ids(client, parsed_flares_dir, time_minutes):
    """Returns IDs of flares from teh temporal test set, not used in individual model training/testing"""
    temporal_test_set_flare_ids = []
    for flare_class in ["B", "<C", ">=C5", "M", "X"]:
        this_flare_class_ids = []
        if "C" in flare_class:
            if "<" in flare_class:
                filter = {'FlareClass': {'$regex': f'^C0|C1|C2|C3|C4'}, 'FlareID': {'$regex': '_0$'}}
            else:
                filter = {'FlareClass': {'$regex': f'^C5|C6|C7|C8|C9'}, 'FlareID': {'$regex': '_0$'}}
        else:
            filter = {'FlareClass': {'$regex': f'^{flare_class}'}, 'FlareID': {'$regex': '_0$'}}
        project = {'FlareID': 1}
        cursor = client['Flares']['NaiveFlares'].find(filter=filter, projection=project)
        for record in cursor:
            this_flare_class_ids.append(record["FlareID"].split("_")[0])
        this_flare_class_ids = pd.DataFrame(np.array(this_flare_class_ids))
        this_flare_class_ids = this_flare_class_ids.sample(frac=0.05, random_state=RANDOM_STATE).reset_index(drop=True)
        for flare_id in this_flare_class_ids.values.tolist():
            temporal_test_set_flare_ids.append(flare_id[0])
    with open(os.path.join("Parsed Flares", parsed_flares_dir, f"Temporal_test_set_flare_ids_{time_minutes}_minutes_since_flare_start.pkl"), "wb") as f:
        pickle.dump(temporal_test_set_flare_ids, f)


def get_tree(results_dir, minutes_since_flare_start, trained=True):

    if trained:
        filename = f"trained_{int(minutes_since_flare_start) - 15}_minutes_since_start"
    else:
        filename = f"untrained_{int(minutes_since_flare_start) - 15}_minutes_since_start"

    with open(os.path.join(results_dir, "Trees", filename), 'rb') as f:
        tree = pickle.load(f)

    return tree


def get_inputs_dict(results_folderpath, run_nickname):

    for root, dirs, files in os.walk(os.path.join(results_folderpath, run_nickname)):
        for file in files:
            if 'inputs' in file:
                with open(os.path.join(root, file), 'rb') as f:
                    inputs = pickle.load(f)
                    return inputs


def flare_c5_or_higher(flux_level):

    if flux_level[0] == "A" or flux_level[0] == "B":
        return False
    elif flux_level[0] == "M" or flux_level[0] == "X":
        return True
    elif flux_level[0] == "C":
        if float(flux_level[1:]) < 5.0:
            return False
        else:
            return True


def is_flare_strong(test_flux_level, threshold_letter_level="C", threshold_number_level=5.0):

    ordered_letter_classes = ["A", "B", "C", "M", "X"]
    threshold_flare_letter_index = ordered_letter_classes.index(threshold_letter_level)

    test_flux_level_letter_index = ordered_letter_classes.index(test_flux_level[0])
    # test flare is weaker
    if test_flux_level_letter_index < threshold_flare_letter_index:
        return False
    # test flare is stronger
    elif test_flux_level_letter_index > threshold_flare_letter_index:
        return True
    # test flare is the same letter class
    if float(test_flux_level[1:]) < float(threshold_number_level):
        return False
    else:
        return True


def assign_is_strong_flare_column(df):

    df["IsFlareClass>=C5"] = (((df.FlareClass.str.startswith("C") == True) &
                               (df.FlareClass.str[1:].astype(float) >= 5.0)) |
                               (df.FlareClass.str.startswith("M") == True) |
                               (df.FlareClass.str.startswith("X") == True))

    return df


def get_temporal_test_set(tree_data, time_minutes, parsed_flares_dir):

    df_temporal_test = pd.DataFrame()

    with open(os.path.join("Parsed Flares", parsed_flares_dir, f"Temporal_test_set_flare_ids_{time_minutes}_minutes_since_flare_start.pkl"), "rb") as f:
        temporal_test_set_flare_ids = pickle.load(f)

    # extract the temporal_test dataset for full time testing first
    # because of seeding these will be the same for all timestamps aside from blacklisted flares
    for flare_id in temporal_test_set_flare_ids:
        tree_data_row = tree_data[tree_data.FlareID == int(flare_id)]
        df_temporal_test = pd.concat([df_temporal_test, tree_data_row], axis=0)
        tree_data = tree_data.drop(index=tree_data_row.index)

    df_temporal_test = df_temporal_test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    temporal_test_x = df_temporal_test.iloc[:, :-1]
    temporal_test_y = df_temporal_test.iloc[:, -1:]

    temporal_test_x_additional_flare_data = pd.concat([temporal_test_x.FlareID, temporal_test_x.FlareClass, temporal_test_x.MinutesToPeak], axis=1)
    temporal_test_x_additional_flare_data = temporal_test_x_additional_flare_data.reset_index()
    temporal_test_x = temporal_test_x.drop(["FlareID"], axis=1)
    temporal_test_x = temporal_test_x.drop(["FlareClass"], axis=1)
    temporal_test_x = temporal_test_x.drop(["MinutesToPeak"], axis=1)

    temporal_test_x_additional_flare_data = assign_is_strong_flare_column(temporal_test_x_additional_flare_data)

    return {"x": temporal_test_x, "y": temporal_test_y, "additional_data": temporal_test_x_additional_flare_data}


def get_stratified_training_and_test_sets(tree_data, train_proportion=0.8):

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # stratify based on the combination of flare class and what max flux was observable
    for idx, flare_mag_letter in enumerate(["B", "C", "M", "X"]):
        for class_id in [0, 1, 2, 3]:  # TODO: Assuming multiclass for now, fix to accommodate binary if desired
            subset = tree_data[(tree_data.FlareClass.str.startswith(flare_mag_letter)) & (tree_data.IsStrongFlare == class_id)]
            subset = subset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            number_of_training_rows = int(subset.shape[0] * train_proportion)
            df_train = pd.concat([df_train, subset.iloc[:number_of_training_rows, :]], axis=0)
            df_test = pd.concat([df_test, subset.iloc[number_of_training_rows:, :]], axis=0)

    df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_x = df_train.iloc[:, :-1]
    train_y = df_train.iloc[:, -1:]
    test_x = df_test.iloc[:, :-1]
    test_y = df_test.iloc[:, -1:]

    train_x_additional_flare_data = pd.concat([train_x.FlareID, train_x.FlareClass, train_x.MinutesToPeak], axis=1)
    train_x_additional_flare_data = train_x_additional_flare_data.reset_index()
    train_x = train_x.drop(["FlareID"], axis=1)
    train_x = train_x.drop(["FlareClass"], axis=1)
    train_x = train_x.drop(["MinutesToPeak"], axis=1)
    test_x_additional_flare_data = pd.concat([test_x.FlareID, test_x.FlareClass, test_x.MinutesToPeak], axis=1)
    test_x_additional_flare_data = test_x_additional_flare_data.reset_index()
    test_x = test_x.drop(["FlareID"], axis=1)
    test_x = test_x.drop(["FlareClass"], axis=1)
    test_x = test_x.drop(["MinutesToPeak"], axis=1)

    train_x_additional_flare_data = assign_is_strong_flare_column(train_x_additional_flare_data)
    test_x_additional_flare_data = assign_is_strong_flare_column(test_x_additional_flare_data)

    return {"train": {"x": train_x, "y": train_y, "additional_data": train_x_additional_flare_data},
            "test": {"x": test_x, "y": test_y, "additional_data": test_x_additional_flare_data}}


def get_training_and_test_sets(tree_data, train_proportion=0.8):

    rows, cols = tree_data.shape
    split = int(rows * train_proportion)
    train = tree_data.iloc[:split, :]
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1:]
    test = tree_data.iloc[split:, :]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1:]

    train_x_additional_flare_data = pd.concat([train_x.FlareID, train_x.FlareClass, train_x.MinutesToPeak], axis=1)
    train_x_additional_flare_data = train_x_additional_flare_data.reset_index()
    train_x = train_x.drop(["FlareID"], axis=1)
    train_x = train_x.drop(["FlareClass"], axis=1)
    train_x = train_x.drop(["MinutesToPeak"], axis=1)
    test_x_additional_flare_data = pd.concat([test_x.FlareID, test_x.FlareClass, test_x.MinutesToPeak], axis=1)
    test_x_additional_flare_data = test_x_additional_flare_data.reset_index()
    test_x = test_x.drop(["FlareID"], axis=1)
    test_x = test_x.drop(["FlareClass"], axis=1)
    test_x = test_x.drop(["MinutesToPeak"], axis=1)

    return train_x, train_x_additional_flare_data, train_y, test_x, test_x_additional_flare_data, test_y


def get_train_and_test_data_from_pkl(minutes_since_start, strong_flare_threshold, use_naive_diffs=True, peak_filtering_minutes=0, stratify=True):

    strong_flare_threshold_letter = strong_flare_threshold[0]
    strong_flare_threshold_number = strong_flare_threshold[1:]
    parsed_flares_dir = f"peak_threshold_minutes_{peak_filtering_minutes}_threshold_{strong_flare_threshold_letter}{strong_flare_threshold_number}"
    if use_naive_diffs:
        parsed_flares_dir += "_naive"

    data_filepath = os.path.join("Parsed Flares", parsed_flares_dir, f"{minutes_since_start}_minutes_since_start.pkl")
    with open(data_filepath, "rb") as f:
        tree_data = pickle.load(f)

    if stratify:
        train_x, train_x_additional_flare_data, train_y, test_x, test_x_additional_flare_data, test_y = get_stratified_training_and_test_sets(tree_data)
    else:
        train_x, train_x_additional_flare_data, train_y, test_x, test_x_additional_flare_data, test_y = get_training_and_test_sets(tree_data)

    return train_x, train_x_additional_flare_data, train_y, test_x, test_x_additional_flare_data, test_y

def linear_interpolation(input_data, minutes_since_start):
    """Linearly interpolates missing data in <input_data>. Uses lookup tables created by imputer.py"""

    for column_name in input_data.columns:
        if column_name != "FlareID" and column_name != "IsStrongFlare":
            col = input_data[column_name]
            if col.isna().sum() != 0:
                lookup_table_path = os.path.join("Interpolations", f"{minutes_since_start - 15}_minutes_since_start", f"{column_name}.pkl")
                with open(lookup_table_path, "rb") as f:
                    lookup_table = pickle.load(f)
                for nan_idx in col[col.isnull()].index.to_list():
                    # The ROWS in training/test data are each flare idx, and correspond to the COLUMNS of the lookup table
                    # The ROWS in the lookup table are the number of minutes since the start of the FITS file
                    if "EmissionMeasure" not in column_name:
                        col[nan_idx] = lookup_table.iloc[minutes_since_start, nan_idx]
                    else:
                        col[nan_idx] = lookup_table.iloc[minutes_since_start, nan_idx] / (10 ** 30)

    return input_data


def impute_variable_data(train_x, test_x, strategy):

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(train_x.values)
    train_x = pd.DataFrame(imp.transform(train_x.values), dtype=np.float64)
    test_x = pd.DataFrame(imp.transform(test_x.values), dtype=np.float64)

    return train_x, test_x


def merge_results_files(results_folderpath, run_nickname):

    all_results = pd.DataFrame()
    for root, dirs, files in os.walk(os.path.join(results_folderpath, run_nickname, "Optimal Tree Hyperparameters")):
        for file in files:
            if "results" in file and file.endswith(".pkl"):
                with open(os.path.join(root, file), "rb") as f:
                    this_time_results = pickle.load(f)
                all_results = pd.concat([all_results, this_time_results])

    return all_results


def get_results_pickle(results_folderpath, run_nickname):

    all_results_filepath = os.path.join(results_folderpath, run_nickname, "results.pkl")
    if os.path.exists(all_results_filepath):
        with open(all_results_filepath, "rb") as f:
            results = pickle.load(f)
    else:
        results = merge_results_files(results_folderpath, run_nickname)
        with open(all_results_filepath, "wb") as f:
            pickle.dump(results, f)

    return results


def create_tree_from_df(grid_search_results, minutes_since_start, max_depth_override=None, ccp_alpha=0.0):
    """Creates a tree from a DataFrame of best parameters for each timestamp"""

    relevant_row = grid_search_results[grid_search_results.minutes_since_start == str(minutes_since_start)]
    t = DecisionTreeClassifier(criterion=relevant_row.criterion.iloc[0],
                               max_depth=int(relevant_row.max_depth.iloc[0]) if max_depth_override is None else max_depth_override,
                               max_features=int(relevant_row.max_features.iloc[0]),
                               min_samples_leaf=int(relevant_row.min_samples_leaf.iloc[0]),
                               min_samples_split=int(relevant_row.min_samples_split.iloc[0]),
                               min_weight_fraction_leaf=float(relevant_row.min_weight_fraction_leaf.iloc[0]),
                               class_weight=relevant_row.class_weight.iloc[0],
                               ccp_alpha=ccp_alpha,
                               random_state=RANDOM_STATE)

    return t


def create_tree(criterion, max_depth, max_features, min_samples_leaf, min_samples_split, min_weight_fraction_leaf, ccp_alpha):
    """Creates a tree from a DataFrame of best parameters for each timestamp"""

    t = DecisionTreeClassifier(criterion=criterion,
                               max_depth=max_depth,
                               max_features=max_features,
                               min_samples_leaf=min_samples_leaf,
                               min_samples_split=min_samples_split,
                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                               class_weight='balanced',
                               ccp_alpha=ccp_alpha,
                               random_state=RANDOM_STATE)

    return t

FORMAL_VARIABLE_NAMES = {"CurrentXRSA": "XRSA Flux",
                         "XRSA1MinuteDifference": "XRSA 1-Minute Difference",
                         "XRSA2MinuteDifference": "XRSA 2-Minute Difference",
                         "XRSA3MinuteDifference": "XRSA 3-Minute Difference",
                         "XRSA4MinuteDifference": "XRSA 4-Minute Difference",
                         "XRSA5MinuteDifference": "XRSA 5-Minute Difference",
                         "CurrentXRSB": "XRSB Flux",
                         "XRSB1MinuteDifference": "XRSB 1-Minute Difference",
                         "XRSB2MinuteDifference": "XRSB 2-Minute Difference",
                         "XRSB3MinuteDifference": "XRSB 3-Minute Difference",
                         "XRSB4MinuteDifference": "XRSB 4-Minute Difference",
                         "XRSB5MinuteDifference": "XRSB 5-Minute Difference",
                         "Temperature": "Temperature",
                         "Temperature1MinuteDifference": "Temperature from 1-Minute XRS Difference",
                         "Temperature2MinuteDifference": "Temperature from 2-Minute XRS Difference",
                         "Temperature3MinuteDifference": "Temperature from 3-Minute XRS Difference",
                         "Temperature4MinuteDifference": "Temperature from 4-Minute XRS Difference",
                         "Temperature5MinuteDifference": "Temperature from 5-Minute XRS Difference",
                         "NaiveTemperature1MinuteDifference": "Temperature 1-Minute Difference",
                         "NaiveTemperature2MinuteDifference": "Temperature 2-Minute Difference",
                         "NaiveTemperature3MinuteDifference": "Temperature 3-Minute Difference",
                         "NaiveTemperature4MinuteDifference": "Temperature 4-Minute Difference",
                         "NaiveTemperature5MinuteDifference": "Temperature 5-Minute Difference",
                         "EmissionMeasure": "Emission Measure",
                         "EmissionMeasure1MinuteDifference": "Emission Measure from 1-Minute XRS Difference",
                         "EmissionMeasure2MinuteDifference": "Emission Measure from 2-Minute XRS Difference",
                         "EmissionMeasure3MinuteDifference": "Emission Measure from 3-Minute XRS Difference",
                         "EmissionMeasure4MinuteDifference": "Emission Measure from 4-Minute XRS Difference",
                         "EmissionMeasure5MinuteDifference": "Emission Measure from 5-Minute XRS Difference",
                         "NaiveEmissionMeasure1MinuteDifference": "Emission Measure 1-Minute Difference",
                         "NaiveEmissionMeasure2MinuteDifference": "Emission Measure 2-Minute Difference",
                         "NaiveEmissionMeasure3MinuteDifference": "Emission Measure 3-Minute Difference",
                         "NaiveEmissionMeasure4MinuteDifference": "Emission Measure 4-Minute Difference",
                         "NaiveEmissionMeasure5MinuteDifference": "Emission Measure 5-Minute Difference",


                         }

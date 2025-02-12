from pymongo import MongoClient
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
import pickle


"""Common utilities for decision trees"""

RANDOM_STATE = 102024
LAUNCH_TIME_MINUTES = 8
OBSERVATION_TIME_MINUTES = 6


def connect_to_flares_db(use_naive=False):
    """Connect to local MongoDB Flares DB"""

    client = MongoClient("mongodb://localhost:27017/")
    flares_db = client["Flares"]
    if use_naive:
        flares_table = flares_db["NaiveFlares"]
    else:
        flares_table = flares_db["Flares"]

    return client, flares_table


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


def get_stratified_training_and_test_sets(tree_data, train_proportion=0.8):

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # stratify based on teh combination of flare class and what max flux was observable
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

    return train_x, train_x_additional_flare_data, train_y, test_x, test_x_additional_flare_data, test_y


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

    # peak_threshold_minutes_-10000_threshold_M1.0_naive
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
                lookup_table_path = os.path.join("Interpolations", f"{column_name}.pkl")
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
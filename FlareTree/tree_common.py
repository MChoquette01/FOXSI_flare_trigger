from pymongo import MongoClient
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
import pickle


"""Common utilities for decision trees"""

RANDOM_STATE = 102024


def connect_to_flares_db():
    """Connect to local MongoDB Flares DB"""

    client = MongoClient("mongodb://localhost:27017/")
    flares_db = client["Flares"]
    flares_table = flares_db["Flares"]

    return client, flares_table


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


def get_training_and_test_sets(tree_data, train_proportion=0.8):

    rows, cols = tree_data.shape
    split = int(rows * train_proportion)
    train = tree_data.iloc[:split, :]
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1:]
    test = tree_data.iloc[split:, :]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1:]

    train_x_flare_ids = train_x.FlareID
    train_x = train_x.drop(["FlareID"], axis=1)
    test_x_flare_ids = test_x.FlareID
    test_x = test_x.drop(["FlareID"], axis=1)

    return train_x, train_x_flare_ids, train_y, test_x, test_x_flare_ids, test_y


def get_train_and_test_data_from_pkl(minutes_since_start, peak_filtering_minutes=0):

    data_filepath = os.path.join("Parsed Flares", f"peak_threshold_minutes_{peak_filtering_minutes}", f"{minutes_since_start}_minutes_since_start.pkl")
    with open(data_filepath, "rb") as f:
        tree_data = pickle.load(f)

    train_x, train_x_flare_ids, train_y, test_x, test_x_flare_ids, test_y = get_training_and_test_sets(tree_data)
    return train_x, train_x_flare_ids, train_y, test_x, test_x_flare_ids, test_y


def impute_variable_data(train_x, test_x):

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_x.values)
    train_x = pd.DataFrame(imp.transform(train_x.values), dtype=np.float64)
    test_x = pd.DataFrame(imp.transform(test_x.values), dtype=np.float64)

    return train_x, test_x


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
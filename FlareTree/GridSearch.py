from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import argparse
import sys
import tree_common as tc
import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.")

"""Run GridSearchCV on flares from MongoDB"""


class Flare:
    def __init__(self, db_entry):
        flare_id_timestamp = db_entry["FlareID"].split("_")
        self.flare_id = flare_id_timestamp[0]
        self.minutes_from_start = int(flare_id_timestamp[1])
        self.flare_class = db_entry["FlareClass"]
        self.is_c5_or_higher = int(tc.flare_c5_or_higher(self.flare_class))
        self.minutes_to_peak = db_entry["MinutesToPeak"]
        self.current_xrsa = db_entry["CurrentXRSA"]
        self.current_xrsb = db_entry["CurrentXRSB"]
        self.xrsa_one_minute_difference = db_entry["XRSA1MinuteDifference"]
        self.xrsa_three_minute_difference = db_entry["XRSA3MinuteDifference"]
        self.xrsa_five_minute_difference = db_entry["XRSA5MinuteDifference"]
        self.xrsb_one_minute_difference = db_entry["XRSB1MinuteDifference"]
        self.xrsb_three_minute_difference = db_entry["XRSB3MinuteDifference"]
        self.xrsb_five_minute_difference = db_entry["XRSB5MinuteDifference"]
        self.temperature = db_entry["Temperature"]
        self.temperature_one_minute_difference = db_entry["Temperature1MinuteDifference"]
        self.temperature_three_minute_difference = db_entry["Temperature3MinuteDifference"]
        self.temperature_five_minute_difference = db_entry["Temperature5MinuteDifference"]
        # raw EM values are too large for float32 (which trees seem to use), scale by 10**-30
        self.emission_measure = db_entry["EmissionMeasure"] / (10 ** 30) if db_entry["EmissionMeasure"] is not None else None
        self.emission_measure_one_minute_difference = db_entry["EmissionMeasure1MinuteDifference"] / (10 ** 30) if db_entry["EmissionMeasure1MinuteDifference"] is not None else None
        self.emission_measure_three_minute_difference = db_entry["EmissionMeasure3MinuteDifference"] / (10 ** 30) if db_entry["EmissionMeasure3MinuteDifference"] is not None else None
        self.emission_measure_five_minute_difference = db_entry["EmissionMeasure5MinuteDifference"] / (10 ** 30) if db_entry["EmissionMeasure5MinuteDifference"] is not None else None
        self.naive_temperature_one_minute_difference = db_entry["NaiveTemperature1MinuteDifference"]
        self.naive_temperature_three_minute_difference = db_entry["NaiveTemperature3MinuteDifference"]
        self.naive_temperature_five_minute_difference = db_entry["NaiveTemperature5MinuteDifference"]
        self.naive_emission_measure_one_minute_difference = db_entry["NaiveEmissionMeasure1MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure1MinuteDifference"] is not None else None
        self.naive_emission_measure_three_minute_difference = db_entry["NaiveEmissionMeasure3MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure3MinuteDifference"] is not None else None
        self.naive_emission_measure_five_minute_difference = db_entry["NaiveEmissionMeasure5MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure5MinuteDifference"] is not None else None
        self.xrsa_remaining = db_entry["XRSARemaining"]
        self.xrsb_remaining = db_entry["XRSBRemaining"]


def grid_search(peak_filtering_threshold_minutes, time_minutes, nan_removal_strategy, scoring_metric, use_naive_diffs, run_nickname):

    # get flares
    client, flares_table = tc.connect_to_flares_db(use_naive=use_naive_diffs)

    parsed_flares = []
    all_entries = flares_table.find({})
    for record in all_entries:
        parsed_flares.append(Flare(record))

    client.close()

    run_nickname = f'{datetime.now().strftime("%d_%m_%Y")}_{run_nickname}'
    if use_naive_diffs:
        parsed_flares_dir = f"peak_threshold_minutes_{peak_filtering_threshold_minutes}_naive"
    else:
        parsed_flares_dir = f"peak_threshold_minutes_{peak_filtering_threshold_minutes}"

    # output folders
    if not os.path.exists(os.path.join("Results", run_nickname)):
        os.makedirs(os.path.join("Results", run_nickname))
    if not os.path.exists(os.path.join("Parsed Flares", parsed_flares_dir)):
        os.makedirs(os.path.join("Parsed Flares", parsed_flares_dir))
    if not os.path.exists(os.path.join("Results", run_nickname, "Tree Graphs")):
        os.mkdir(os.path.join("Results", run_nickname, "Tree Graphs"))
    if not os.path.exists(os.path.join("Results", run_nickname, "Optimal Tree Hyperparameters")):
        os.mkdir(os.path.join("Results", run_nickname, "Optimal Tree Hyperparameters"))

    results = []  # store best params for each timestamp
    out_path = os.path.join("Parsed Flares", parsed_flares_dir, f"{time_minutes}_minutes_since_start.pkl")
    if not os.path.exists(out_path):
        # open naive differences DF
        with open(os.path.join("Naive Differences", "naive_differences.pkl"), 'rb') as f:
            naive_differences = pickle.load(f)
        tree_data = []
        for flare in tqdm(parsed_flares, desc=f"Parsing flares ({time_minutes} minutes since start)..."):
            if flare.minutes_from_start == time_minutes and flare.minutes_to_peak > peak_filtering_threshold_minutes:
                tree_data.append([flare.flare_id,
                                  flare.current_xrsa,
                                  flare.xrsa_one_minute_difference,
                                  flare.xrsa_three_minute_difference,
                                  flare.xrsa_five_minute_difference,
                                  flare.current_xrsb,
                                  flare.xrsb_one_minute_difference,
                                  flare.xrsb_three_minute_difference,
                                  flare.xrsb_five_minute_difference,
                                  flare.temperature,
                                  flare.temperature_one_minute_difference,
                                  flare.temperature_three_minute_difference,
                                  flare.temperature_five_minute_difference,
                                  flare.emission_measure,
                                  flare.emission_measure_one_minute_difference,
                                  flare.emission_measure_three_minute_difference,
                                  flare.emission_measure_five_minute_difference,
                                  flare.naive_temperature_one_minute_difference,
                                  flare.naive_temperature_three_minute_difference,
                                  flare.naive_temperature_five_minute_difference,
                                  flare.naive_emission_measure_one_minute_difference if flare.naive_emission_measure_one_minute_difference is not None else None,
                                  flare.naive_emission_measure_three_minute_difference if flare.naive_emission_measure_three_minute_difference is not None else None,
                                  flare.naive_emission_measure_five_minute_difference if flare.naive_emission_measure_five_minute_difference is not None else None,
                                  flare.is_c5_or_higher]
                                 )

        tree_data = pd.DataFrame(np.array(tree_data), dtype=np.float64)
        tree_data.columns = ["FlareID", "CurrentXRSA", "XRSA1MinuteDifference", "XRSA3MinuteDifference",
                             "XRSA5MinuteDifference", "CurrentXRSB", "XRSB1MinuteDifference", "XRSB3MinuteDifference",
                             "XRSB5MinuteDifference", "Temperature", "Temperature1MinuteDifference",
                             "Temperature3MinuteDifference", "Temperature5MinuteDifference", "EmissionMeasure",
                             "EmissionMeasure1MinuteDifference", "EmissionMeasure3MinuteDifference",
                             "EmissionMeasure5MinuteDifference", "NaiveTemperature1MinuteDifference",
                             "NaiveTemperature3MinuteDifference", "NaiveTemperature5MinuteDifference",
                             "NaiveEmissionMeasure1MinuteDifference", "NaiveEmissionMeasure3MinuteDifference",
                             "NaiveEmissionMeasure5MinuteDifference", "IsC5OrHigher"]

        with open(out_path, "wb") as f:
            pickle.dump(tree_data, f)

    with open(out_path, "rb") as f:
        tree_data = pickle.load(f)

    if nan_removal_strategy == "linear_interpolation":
        tree_data = tc.linear_interpolation(tree_data, time_minutes)

    train_x, _, train_y, test_x, _, test_y = tc.get_training_and_test_sets(tree_data)

    if nan_removal_strategy != "linear_interpolation":
        train_x, test_x = tc.impute_variable_data(train_x, test_x, nan_removal_strategy)

    train_x.columns = list(tree_data.columns)[1:-1]

    t = DecisionTreeClassifier(random_state=tc.RANDOM_STATE)

    # Test parameters, should run in a few minutes for ~30 trees
    params = {"criterion": ["gini", "entropy"],
              "max_depth": [x for x in range(12, 15)],
              "min_samples_split": [x for x in range(4, 5)],
              "min_samples_leaf": [x for x in range(1, 2)],
              "min_weight_fraction_leaf": [x / 10 for x in range(2)],
              "max_features": [x for x in range(2, 4)],
              "class_weight": ["balanced"]}

    # Real deal parameters
    # params = {"criterion": ['gini', 'entropy'],
    #           "max_depth": [x for x in range(10, 21)],
    #           "min_samples_split": [x * 5 for x in range(1, 11)],
    #           "min_samples_leaf": [x * 5 for x in range(1, 11)],
    #           "min_weight_fraction_leaf": [x / 10 for x in range(2)],
    #           "max_features": [x for x in range(5, 17)],
    #           "class_weight": ["balanced"]}

    # fun with custom scoring functions!
    def false_positive_scorer(y_true, y_predicted):
        cm = confusion_matrix(y_true, y_predicted)
        tn, fp, fn, tp = cm.ravel()
        return fp / (tn + fp)

    fpr_scorer = make_scorer(false_positive_scorer, greater_is_better=False)
    if scoring_metric == "false_positive_rate":
        scoring_metric = fpr_scorer
    gs = GridSearchCV(t, params, scoring=scoring_metric)
    gs.fit(train_x.values, train_y.values)
    best_params = gs.best_params_
    # print(best_params)
    # print(gs.best_score_)
    # initialize the best tree
    t = DecisionTreeClassifier(criterion=best_params["criterion"],
                               max_depth=best_params["max_depth"],
                               max_features=best_params["max_features"],
                               min_samples_leaf=best_params["min_samples_leaf"],
                               min_samples_split=best_params["min_samples_split"],
                               min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
                               class_weight=best_params["class_weight"],
                               random_state=tc.RANDOM_STATE)
    t.fit(train_x.values, train_y.values)
    test_predictions = t.predict(test_x.values)

    # create confusion matrices (or rather, the related stats) for training and test sets
    test_y_reshaped = test_y.to_numpy().reshape((np.shape(test_y)[0],))
    # ConfusionMatrixDisplay.from_predictions(test_y, predictions, display_labels=["Small Flare", "Big Flare"])
    # plt.show()
    test_cm = confusion_matrix(test_y, test_predictions)
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()

    test_tpr = test_tp/(test_tp+test_fn)
    test_fpr = test_fp/(test_tp+test_fp)
    test_tss = test_tpr - test_fpr
    test_precision = precision_score(test_y, test_predictions)
    test_recall = recall_score(test_y, test_predictions)
    test_f1 = f1_score(test_y, test_predictions)

    train_predictions = t.predict(train_x.values)
    train_y_reshaped = train_y.to_numpy().reshape((np.shape(train_y)[0],))
    train_cm = confusion_matrix(train_y, train_predictions)
    train_tn, train_fp, train_fn, train_tp = train_cm.ravel()

    train_tpr = train_tp/(train_tp+train_fn)
    train_fpr = train_fp/(train_tp+train_fp)
    train_tss = train_tpr - train_fpr
    train_precision = precision_score(train_y, train_predictions)
    train_recall = recall_score(train_y, train_predictions)
    train_f1 = f1_score(train_y, train_predictions)

    results.append([time_minutes,
                    tree_data.shape[0],
                    tree_data[tree_data.IsC5OrHigher == 0].shape[0],
                    tree_data[tree_data.IsC5OrHigher == 1].shape[0],
                    best_params["criterion"],
                    int(best_params["max_depth"]),
                    int(best_params["max_features"]),
                    int(best_params["min_samples_leaf"]),
                    int(best_params["min_samples_split"]),
                    best_params["min_weight_fraction_leaf"],
                    best_params["class_weight"],
                    test_precision,
                    train_precision,
                    test_recall,
                    train_recall,
                    test_f1,
                    train_f1,
                    test_tpr,
                    train_tpr,
                    test_fpr,
                    train_fpr,
                    test_tss,
                    train_tss])

    # plot best tree structure
    plt.figure(figsize=(50, 28))  # big, so rectangles don't overlap
    plot_tree(t, feature_names=train_x.columns, class_names=["<C5", ">=C5"], filled=True, proportion=True, rounded=True, precision=9, fontsize=10)
    graph_out_path = os.path.join("Results", run_nickname, "Tree Graphs", f"{time_minutes}_minutes_since_start_tree.png")
    plt.savefig(graph_out_path)

    results = pd.DataFrame(np.array(results))
    results.columns = ["minutes_since_start", "total_number_of_flares", "number_of_<C5_flares", "number_of_>=C5_flares",
                       "criterion", "max_depth", "max_features", "min_samples_leaf", "min_samples_split",
                       "min_weight_fraction_leaf", "class_weight", "test_precision", "train_precision", "test_recall",
                       "train_recall", "test_f1", "train_f1", "test_tpr", "train_tpr", "test_fpr", "train_fpr", "test_tss",
                       "train_tss"]

    with open(os.path.join("Results", run_nickname, "Optimal Tree Hyperparameters", f"results_{time_minutes}.pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join("Results", run_nickname, "Optimal Tree Hyperparameters", f"grid_{time_minutes}.pkl"), "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":

    # running from MSI/Slurm, or CMD line, if you really want to
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", type=int, help="Removes all flares from dataset that are greater than X minutes from their peak")
        parser.add_argument("-s", type=int, help="Start time (from the start of the FITS file, 0 indexed) in minutes to build models for")
        parser.add_argument("-i", type=str, help="Method to replace NaN values. Either a Pandas interpolate() strategy or 'linear_interpolation'")
        parser.add_argument("-m", type=str, help="Sklearn scoring metric to use or 'false_positive_rate'")
        parser.add_argument('--naive', action='store_true', help="Use to parse flares with naive temp/EM differences")
        parser.add_argument('--no-naive', dest='naive', action='store_false')
        parser.set_defaults(add_naive=False)
        parser.add_argument("-n", type=str, help="Run nickname - make sure it's unique!")
        args = parser.parse_args()
        grid_search(peak_filtering_threshold_minutes=args.t,
                    time_minutes=args.s,
                    nan_removal_strategy=args.i,
                    scoring_metric=args.m,
                    use_naive_diffs=args.naive,
                    run_nickname=args.n)

    # run here
    else:
        peak_filtering_threshold_minutes = 0
        time_minutes = 25
        nan_removal_strategy = "linear_interpolation"
        scoring_metric = "f1"
        use_naive_diffs = True
        run_nickname = "naivedifftest"
        grid_search(peak_filtering_threshold_minutes,
                    time_minutes,
                    nan_removal_strategy,
                    scoring_metric,
                    use_naive_diffs,
                    run_nickname)

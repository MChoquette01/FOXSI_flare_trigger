from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import math
from tqdm import tqdm
import re
import argparse
import sys
import tree_common as tc
import warnings
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.")

"""Run GridSearchCV on flares from MongoDB"""


def graph_feature_importance(output_folder, t, minutes_since_start, train_x, run_nickname):
    """Create a bar chart showing tree feature importance"""

    features_importances = t.feature_importances_
    f_i = []
    nans = []
    for idx, feature_importance in enumerate(features_importances):
        if not math.isnan(feature_importance):
            f_i.append([train_x.columns[idx], feature_importance])
        else:
            nans.append([train_x.columns[idx], 0.1])
    f_i = sorted(f_i, key=lambda x: x[1])
    plt.figure(figsize=(16, 9))
    plt.barh([x for x in range(len(f_i))], [x[1] for x in f_i])
    plt.barh([x + len(f_i) for x in range(len(nans))], [x[1] for x in nans], color="red")
    plt.yticks(ticks=[x for x in range(len(f_i) + len(nans))], labels=[x[0] for x in f_i + nans], fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Normalized Total Reduction of Split Criteria", fontsize=20)
    plt.ylabel("Feature", fontsize=20)
    plt.title(f"Feature Importance: {minutes_since_start - 15} minutes since start", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "Results", run_nickname, "Feature Importance", f"FeatureImportance_{minutes_since_start}_minutes_since_start.png"))
    # plt.show()


def graph_confusion_matrices(output_folder, train_y, train_predictions, test_y, test_predictions, run_nickname, time_minutes, strong_flare_threshold):

    plt.clf()
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(25)
    fig.set_figheight(14)
    ax[0].set_title("Training", fontsize=24)
    ax[1].set_title("Test", fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=22)
    ax[1].tick_params(axis='both', which='major', labelsize=22)
    ax[0].set_xlabel(["Predicted Label"], fontsize=24)
    ax[1].set_xlabel(["Predicted Label"], fontsize=24)
    ax[0].set_ylabel(["True Label"], fontsize=24)
    ax[1].set_ylabel(["True Label"], fontsize=24)
    plt.rcParams.update({'font.size': 26})

    if strong_flare_threshold is None:  # multiclass
        display_labels = [f"< C5", f">= C5", f"M", f"X"]
    else:  # binary
        display_labels = [f"< {strong_flare_threshold}", f">= {strong_flare_threshold}"]

    ConfusionMatrixDisplay.from_predictions(train_y, train_predictions, display_labels=display_labels).plot(ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(test_y, test_predictions, display_labels=display_labels).plot(ax=ax[1])
    fig.suptitle(f"Flares {time_minutes - 15} minutes since start", fontsize=24)
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, "Results", run_nickname, "Confusion Matrices", f"ConfusionMatrices_{time_minutes}_minutes_since_start.png"))
    # plt.show()


def get_confusion_matrix_stats(cm):
    tp = np.diag(cm)

    results = {"Precision": [],
               "Recall": [],
               "F1": [],
               "FPR": [],
               "TSS": []}

    for col_idx in range(cm.shape[0]):
        fp = np.sum(cm[:, col_idx]) - tp[col_idx]
        fn = np.sum(cm[col_idx, :]) - tp[col_idx]
        tn = cm.sum() - fp - fn - tp.sum()
        results["Precision"].append(tp[col_idx] / (tp[col_idx] + fp))
        results["Recall"].append(tp[col_idx] / (tp[col_idx] + fn))
        results["F1"].append((2 * results["Precision"][col_idx] * results["Recall"][col_idx]) / (
                    results["Precision"][col_idx] + results["Recall"][col_idx]))
        results["FPR"].append(fp / (fp + tn))
        results["TSS"].append(results["Recall"][col_idx] - results["FPR"][col_idx])

    return results


def make_dir_safe(folder_path):
    """MSI sometimes fails on folder creation when GridSearch.py is being run in parallel"""
    try:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    except FileExistsError:
        pass


def get_max_flux_level_in_observation_time(multiclass, xrsbs, flare_id):
    """0: < C5; 1: <= C5 x < M1; 2: M1 <= x < X0.0; 3: >= X0.0"""

    flare_subset = xrsbs[xrsbs.FlareID == flare_id]
    if multiclass:
        if any(flare_subset.XRSB >= 1*10**-4):  # >= X
            return 3
        elif any(flare_subset.XRSB >= 1*10**-5):  # >= M
            return 2
        elif any(flare_subset.XRSB >= 5*10**-6):  # >= C5
            return 1
        return 0  # < C5

    else: #  >= C5 or not only
        if any(flare_subset.XRSB >= 5*10**-6):  # >= C5
            return 1
        return 0  # < C5


class Flare:
    def __init__(self, db_entry, use_naive_diffs, xrsbs, multiclass):
        flare_id_timestamp = db_entry["FlareID"].split("_")
        self.flare_id = flare_id_timestamp[0]
        self.minutes_from_start = int(flare_id_timestamp[1])
        self.flare_class = db_entry["FlareClass"]
        self.is_strong_flare = get_max_flux_level_in_observation_time(multiclass, xrsbs, self.flare_id)
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
        if use_naive_diffs:
            self.naive_temperature_one_minute_difference = db_entry["NaiveTemperature1MinuteDifference"]
            self.naive_temperature_three_minute_difference = db_entry["NaiveTemperature3MinuteDifference"]
            self.naive_temperature_five_minute_difference = db_entry["NaiveTemperature5MinuteDifference"]
            self.naive_emission_measure_one_minute_difference = db_entry["NaiveEmissionMeasure1MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure1MinuteDifference"] is not None else None
            self.naive_emission_measure_three_minute_difference = db_entry["NaiveEmissionMeasure3MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure3MinuteDifference"] is not None else None
            self.naive_emission_measure_five_minute_difference = db_entry["NaiveEmissionMeasure5MinuteDifference"] / (10 ** 30) if db_entry["NaiveEmissionMeasure5MinuteDifference"] is not None else None
        self.xrsa_remaining = db_entry["XRSARemaining"]
        self.xrsb_remaining = db_entry["XRSBRemaining"]


def grid_search(peak_filtering_threshold_minutes, time_minutes, strong_flare_threshold, nan_removal_strategy, scoring_metric, use_naive_diffs, output_folder, run_nickname, model_type, multiclass, debug_mode, stratify=True):

    # save for reproducibility
    inputs = {"peak_filtering_threshold_minutes": peak_filtering_threshold_minutes,
              "time_minutes": time_minutes,
              "strong_flare_threshold": strong_flare_threshold,
              "nan_removal_strategy": nan_removal_strategy,
              "scoring_metric": scoring_metric,
              "use_naive_diffs": use_naive_diffs,
              "output_folder": output_folder,
              "multiclass": multiclass,
              "debug_mode": debug_mode,
              "model_type": model_type,
              "run_nickname": run_nickname,
              "stratify": stratify}

    strong_flare_threshold_letter = strong_flare_threshold[0]
    strong_flare_threshold_number = strong_flare_threshold[1:]

    run_nickname = f'{datetime.now().strftime("%Y_%m_%d")}_{run_nickname}'

    parsed_flares_dir = f"peak_threshold_minutes_{peak_filtering_threshold_minutes}"
    if multiclass:
        parsed_flares_dir += "_multiclass"
    else:
        parsed_flares_dir += f"_threshold_{strong_flare_threshold_letter}{strong_flare_threshold_number}"
    if use_naive_diffs:
        parsed_flares_dir += "_naive"

    # output folders
    make_dir_safe(os.path.join(output_folder, "Parsed Flares"))
    make_dir_safe(os.path.join(output_folder, "Parsed Flares", parsed_flares_dir))
    make_dir_safe(os.path.join(output_folder, "Results"))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname, "Trees"))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname, "Feature Importance"))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname, "Confusion Matrices"))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname, "Tree Graphs"))
    make_dir_safe(os.path.join(output_folder, "Results", run_nickname, "Optimal Tree Hyperparameters"))

    with open(os.path.join(output_folder, "Results", run_nickname, "Optimal Tree Hyperparameters", f"inputs_{time_minutes}.pkl"), "wb") as f:
        pickle.dump(inputs, f)

    results = []  # store best params for each timestamp
    out_path = os.path.join("Parsed Flares", parsed_flares_dir, f"{time_minutes}_minutes_since_start.pkl")
    if not os.path.exists(out_path):
        # get flares
        client, flares_table = tc.connect_to_flares_db(use_naive=use_naive_diffs)

        # get XRSB table
        regex_string = ""
        for timestamp in range(time_minutes + tc.LAUNCH_TIME_MINUTES, time_minutes + tc.LAUNCH_TIME_MINUTES + tc.OBSERVATION_TIME_MINUTES):
            regex_string += f"_{timestamp}|"
        regex_string = regex_string[:-1]

        xrsbs_during_observation = []
        filter = {'FlareID': {'$regex': regex_string}}
        project = {'CurrentXRSB': 1, 'FlareID': 1}
        sort = list({}.items())
        limit = 0
        result = client['Flares']['NaiveFlares'].find(filter=filter, projection=project, sort=sort, limit=limit)
        for record in result:
            flare_id, timestamp = record["FlareID"].split("_")
            if int(flare_id) not in tc.BLACKLISTED_FLARE_IDS:
                xrsbs_during_observation.append([flare_id, timestamp, record["CurrentXRSB"]])

        xrsbs_during_observation = pd.DataFrame(np.array(xrsbs_during_observation))
        xrsbs_during_observation.columns = ["FlareID", "Timestamp", "XRSB"]

        parsed_flares = []
        all_entries = flares_table.find({'FlareID': {'$regex': f'_{time_minutes}$'}})
        for record in tqdm(all_entries, desc=f"Creating flare objects ({time_minutes} minutes since start)..."):
            if int(record["FlareID"].split("_")[0]) not in tc.BLACKLISTED_FLARE_IDS:
                parsed_flares.append(Flare(record, use_naive_diffs, xrsbs_during_observation, multiclass))

        client.close()

        tree_data = []
        flare_classes = []  # handle separately because these are strings, not ints
        for flare in tqdm(parsed_flares, desc=f"Parsing flares ({time_minutes} minutes since start)..."):
            if flare.minutes_from_start == time_minutes and flare.minutes_to_peak > peak_filtering_threshold_minutes:
                flare_classes.append([flare.flare_class])
                if use_naive_diffs:
                    tree_data.append([flare.flare_id,
                                      flare.minutes_to_peak,
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
                                      flare.is_strong_flare])
                else:
                    tree_data.append([flare.flare_id,
                                      flare.minutes_to_peak,
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
                                      flare.is_strong_flare])

        tree_data = pd.DataFrame(np.array(tree_data), dtype=np.float64)
        targets = tree_data.iloc[:, -1:]
        tree_data.drop(tree_data.columns[-1], axis=1, inplace=True)
        # flare class is not numeric...handle separately
        flare_classes = pd.DataFrame(np.array(flare_classes))
        tree_data = pd.concat([tree_data, flare_classes, targets], axis=1)

        if use_naive_diffs:
            tree_data.columns = ["FlareID", "MinutesToPeak", "CurrentXRSA", "XRSA1MinuteDifference",
                                 "XRSA3MinuteDifference", "XRSA5MinuteDifference", "CurrentXRSB",
                                 "XRSB1MinuteDifference", "XRSB3MinuteDifference", "XRSB5MinuteDifference",
                                 "Temperature", "Temperature1MinuteDifference", "Temperature3MinuteDifference",
                                 "Temperature5MinuteDifference", "EmissionMeasure", "EmissionMeasure1MinuteDifference",
                                 "EmissionMeasure3MinuteDifference", "EmissionMeasure5MinuteDifference",
                                 "NaiveTemperature1MinuteDifference", "NaiveTemperature3MinuteDifference",
                                 "NaiveTemperature5MinuteDifference", "NaiveEmissionMeasure1MinuteDifference",
                                 "NaiveEmissionMeasure3MinuteDifference", "NaiveEmissionMeasure5MinuteDifference",
                                 "FlareClass", "IsStrongFlare"]
        else:
            tree_data.columns = ["FlareID", "MinutesToPeak", "CurrentXRSA", "XRSA1MinuteDifference",
                                 "XRSA3MinuteDifference", "XRSA5MinuteDifference", "CurrentXRSB",
                                 "XRSB1MinuteDifference", "XRSB3MinuteDifference", "XRSB5MinuteDifference",
                                 "Temperature", "Temperature1MinuteDifference", "Temperature3MinuteDifference",
                                 "Temperature5MinuteDifference", "EmissionMeasure", "EmissionMeasure1MinuteDifference",
                                 "EmissionMeasure3MinuteDifference", "EmissionMeasure5MinuteDifference",
                                 "FlareClass", "IsStrongFlare"]

        with open(out_path, "wb") as f:
            pickle.dump(tree_data, f)

    with open(out_path, "rb") as f:
        tree_data = pickle.load(f)

    if nan_removal_strategy == "linear_interpolation":
        tree_data = tc.linear_interpolation(tree_data, time_minutes)

    if stratify:
        train_x, train_x_additional_data, train_y, test_x, test_x_additional_data, test_y = tc.get_stratified_training_and_test_sets(tree_data)
    else:
        train_x, train_x_additional_data, train_y, test_x, test_x_additional_data, test_y = tc.get_training_and_test_sets(tree_data)

    if nan_removal_strategy != "linear_interpolation":
        train_x, test_x = tc.impute_variable_data(train_x, test_x, nan_removal_strategy)

    train_x.columns = list(tree_data.columns)[2:-2]

    if model_type == "Tree":
        model = DecisionTreeClassifier(random_state=tc.RANDOM_STATE)
        if debug_mode:
            # Test parameters, should run in a few minutes for ~30 trees
            params = {"criterion": ["gini", "entropy"],
                      "max_depth": [x for x in range(12, 15)],
                      "min_samples_split": [x for x in range(4, 5)],
                      "min_samples_leaf": [x for x in range(1, 2)],
                      "min_weight_fraction_leaf": [x / 10 for x in range(2)],
                      "max_features": [x for x in range(5, 17)],
                      "class_weight": ["balanced"]}
        else:
            # Real deal parameters
            params = {"criterion": ['gini', 'entropy'],
                      "max_depth": [x for x in range(10, 21)],
                      "min_samples_split": [x * 5 for x in range(1, 11)],
                      "min_samples_leaf": [x * 5 for x in range(1, 11)],
                      "min_weight_fraction_leaf": [x / 10 for x in range(2)],
                      "max_features": [x for x in range(5, 23)],
                      "class_weight": ["balanced"]}

    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=tc.RANDOM_STATE)
        if debug_mode:
            params = {"n_estimators": [100],
                      "criterion": ["gini"],
                      "max_depth": [x for x in range(13, 15)],
                      "min_samples_split": [x for x in range(4, 5)],
                      "min_samples_leaf": [x for x in range(1, 2)],
                      "min_weight_fraction_leaf": [x / 10 for x in range(2)],
                      "max_features": [x for x in range(15, 17)],
                      "class_weight": ["balanced"]}
        else:
            # Real deal parameters
            params = {"n_estimators": [50 * x for x in range(2, 7)],
                      "criterion": ["gini", "entropy"],
                      "max_depth": [x for x in range(12, 15)],
                      "min_samples_split": [x for x in range(10, 20)],
                      "min_samples_leaf": [x for x in range(10, 20)],
                      "min_weight_fraction_leaf": [x / 10 for x in range(2)],
                      "max_features": [x for x in range(5, 23)],
                      "class_weight": ["balanced"]}

    elif model_type == "Gradient Boosted Tree":
        model = GradientBoostingClassifier(random_state=tc.RANDOM_STATE)
        if debug_mode:
            # Test parameters, should run in a few minutes for ~30 trees
            params = {"loss": ["log_loss"],
                      "learning_rate": [0.1],
                      "n_estimators": [100, 150],
                      "criterion": ["friedman_mse"],
                      "max_depth": [15],
                      "min_samples_split": [5],
                      "min_samples_leaf": [10],
                      "min_weight_fraction_leaf": [0.1],
                      "max_features": [15]
                      }
        else:
            # Real deal parameters
            params = {"loss": ["log_loss"],
                      "learning_rate": [0.05, 0.1, 0.15],
                      "n_estimators": [50 * x for x in range(2, 7)],
                      "criterion": ["friedman_mse"],
                      "max_depth": [x for x in range(12, 15)],
                      "min_samples_split": [x for x in range(10, 20)],
                      "min_samples_leaf": [x for x in range(10, 20)],
                      "min_weight_fraction_leaf": [x / 10 for x in range(2)],
                      "max_features": [x for x in range(5, 23)]}

    # fun with custom scoring functions!
    def false_positive_scorer(y_true, y_predicted):
        cm = confusion_matrix(y_true, y_predicted)
        tn, fp, fn, tp = cm.ravel()
        return fp / (tn + fp)

    def precision_scorer(y_true, y_predicted):
        return precision_score(y_true, y_predicted, average='macro', zero_division=0.0)

    def f1_scorer(y_true, y_predicted):
        return f1_score(y_true, y_predicted, average='macro', zero_division=0.0)

    def balanced_accuracy_scorer(y_true, y_predicted):
        return balanced_accuracy_score(y_true, y_predicted)

    fpr_scorer = make_scorer(false_positive_scorer, greater_is_better=False)
    p_scorer = make_scorer(precision_scorer, greater_is_better=True)
    f1_scorer = make_scorer(f1_scorer, greater_is_better=True)
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_scorer, greater_is_better=True)
    if scoring_metric == "false_positive_rate":
        scoring_metric = fpr_scorer
    elif scoring_metric == "precision":
        scoring_metric = p_scorer
    elif scoring_metric == "f1":
        scoring_metric = f1_scorer
    elif scoring_metric == "balanced_accuracy":
        scoring_metric = balanced_accuracy_scorer

    gs = GridSearchCV(model, params, scoring=scoring_metric, n_jobs=3)
    gs.fit(train_x.values, train_y.values.ravel())
    best_params = gs.best_params_
    # print(best_params)
    # print(gs.best_score_)
    # initialize the best tree
    if model_type == "Tree":
        final_model = DecisionTreeClassifier(criterion=best_params["criterion"],
                                             max_depth=best_params["max_depth"],
                                             max_features=best_params["max_features"],
                                             min_samples_leaf=best_params["min_samples_leaf"],
                                             min_samples_split=best_params["min_samples_split"],
                                             min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
                                             class_weight=best_params["class_weight"],
                                             random_state=tc.RANDOM_STATE)
    elif model_type == "Random Forest":
        final_model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                                             criterion=best_params["criterion"],
                                             max_depth=best_params["max_depth"],
                                             max_features=best_params["max_features"],
                                             min_samples_leaf=best_params["min_samples_leaf"],
                                             min_samples_split=best_params["min_samples_split"],
                                             min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
                                             class_weight=best_params["class_weight"],
                                             random_state=tc.RANDOM_STATE)

    elif model_type == "Gradient Boosted Tree":
        final_model = GradientBoostingClassifier(loss=best_params["loss"],
                                                 learning_rate=best_params["learning_rate"],
                                                 n_estimators=best_params["n_estimators"],
                                                 criterion=best_params["criterion"],
                                                 max_depth=best_params["max_depth"],
                                                 max_features=best_params["max_features"],
                                                 min_samples_leaf=best_params["min_samples_leaf"],
                                                 min_samples_split=best_params["min_samples_split"],
                                                 min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
                                                 random_state=tc.RANDOM_STATE)

    tree_path = os.path.join(output_folder, "Results", run_nickname, "Trees", f"untrained_{time_minutes - 15}_minutes_since_start")
    with open(tree_path, 'wb') as f:
        pickle.dump(final_model, f)
    final_model.fit(train_x.values, train_y.values.ravel())
    with open(tree_path.replace('untrained', 'trained'), 'wb') as f:
        pickle.dump(final_model, f)
    test_predictions = final_model.predict(test_x.values)

    # create confusion matrices (or rather, the related stats) for training and test sets
    test_y_reshaped = test_y.to_numpy().reshape((np.shape(test_y)[0],))
    # ConfusionMatrixDisplay.from_predictions(test_y, predictions, display_labels=["Small Flare", "Big Flare"])
    # plt.show()
    test_cm = confusion_matrix(test_y, test_predictions)
    test_scores = get_confusion_matrix_stats(test_cm)

    train_predictions = final_model.predict(train_x.values)
    train_y_reshaped = train_y.to_numpy().reshape((np.shape(train_y)[0],))
    train_cm = confusion_matrix(train_y, train_predictions)
    train_scores = get_confusion_matrix_stats(train_cm)

    graph_confusion_matrices(output_folder, train_y, train_predictions, test_y, test_predictions, run_nickname, time_minutes, strong_flare_threshold=strong_flare_threshold if multiclass is False else None)
    graph_feature_importance(output_folder, final_model, time_minutes, train_x, run_nickname)

    results.append([time_minutes,
                    tree_data.shape[0],
                    tree_data[tree_data.IsStrongFlare == 0].shape[0],
                    tree_data[tree_data.IsStrongFlare == 1].shape[0],
                    best_params["criterion"],
                    int(best_params["max_depth"]),
                    int(best_params["max_features"]),
                    int(best_params["min_samples_leaf"]),
                    int(best_params["min_samples_split"]),
                    best_params["min_weight_fraction_leaf"],
                    best_params["class_weight"] if "class_weight" in list(best_params.keys()) else None,
                    sum(test_scores["Precision"]) / 4,
                    sum(train_scores["Precision"]) / 4,
                    sum(test_scores["Recall"]) / 4,
                    sum(train_scores["Recall"]) / 4,
                    sum(test_scores["F1"]) / 4,
                    sum(train_scores["F1"]) / 4,
                    sum(test_scores["Recall"]) / 4,  # Labelled as TPR
                    sum(train_scores["Recall"]) / 4,  # Labelled as TPR
                    sum(test_scores["FPR"]) / 4,
                    sum(train_scores["FPR"]) / 4,
                    sum(test_scores["TSS"]) / 4,
                    sum(train_scores["TSS"]) / 4,])

    # plot best tree structure
    plt.figure(figsize=(50, 28))  # big, so rectangles don't overlap
    if model_type == "Tree":
        plot_tree(final_model, feature_names=train_x.columns, class_names=[f"< C5", f"<= C5 x < M0", f"<= M0 x < X0", f">= X0"], filled=True, proportion=True, rounded=True, precision=9, fontsize=10)
    graph_out_path = os.path.join(output_folder, "Results", run_nickname, "Tree Graphs", f"{time_minutes}_minutes_since_start_tree.png")
    plt.savefig(graph_out_path)

    results = pd.DataFrame(np.array(results))
    results.columns = ["minutes_since_start", "total_number_of_flares", "number_of_weak_flares", "number_of_strong_flares",
                       "criterion", "max_depth", "max_features", "min_samples_leaf", "min_samples_split",
                       "min_weight_fraction_leaf", "class_weight", "test_precision", "train_precision", "test_recall",
                       "train_recall", "test_f1", "train_f1", "test_tpr", "train_tpr", "test_fpr", "train_fpr", "test_tss",
                       "train_tss"]

    with open(os.path.join(output_folder, "Results", run_nickname, "Optimal Tree Hyperparameters", f"results_{time_minutes}.pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join(output_folder, "Results", run_nickname, "Optimal Tree Hyperparameters", f"grid_{time_minutes}.pkl"), "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":

    # running from MSI/Slurm, or CMD line, if you really want to
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", type=int, help="Removes all flares from dataset that are greater than X minutes from their peak")
        parser.add_argument("-s", type=int, help="Start time (from the start of the FITS file, 0 indexed) in minutes to build models for")
        parser.add_argument("-f", type=str, help="Inclusively lower bound on what is considered a strong flare (positive class)")
        parser.add_argument("-i", type=str, help="Method to replace NaN values. Either a Pandas interpolate() strategy or 'linear_interpolation'")
        parser.add_argument("-m", type=str, help="Sklearn scoring metric to use or 'false_positive_rate'")
        parser.add_argument("-o", type=str, help="Output folder. A 'Results' folder will be created inside")
        parser.add_argument("-p", type=str, help="Model type to use - 'Tree', 'Random Forest' or 'Gradient Boosted Tree'")
        parser.add_argument("-n", type=str, help="Run nickname - make sure it's unique!")
        parser.add_argument('--multiclass', action='store_true', help="Enables 4 class output instead of 2")
        parser.add_argument('--no-multiclass', dest='multiclass', action='store_false')
        parser.set_defaults(debug=True)
        parser.add_argument('--debug', action='store_true', help="Use to test a quick grid to get results quicker")
        parser.add_argument('--no-debug', dest='debug', action='store_false')
        parser.set_defaults(debug=False)
        parser.add_argument('--naive', action='store_true', help="Use to parse flares with naive temp/EM differences")
        parser.add_argument('--no-naive', dest='naive', action='store_false')
        parser.set_defaults(add_naive=False)
        args = parser.parse_args()
        grid_search(peak_filtering_threshold_minutes=args.t,
                    time_minutes=args.s,
                    strong_flare_threshold=args.f,
                    nan_removal_strategy=args.i,
                    scoring_metric=args.m,
                    output_folder=args.o,
                    run_nickname=args.n,
                    multiclass=args.multiclass,
                    model_type=args.p,
                    use_naive_diffs=args.naive,
                    debug_mode=args.debug)

    # run here
    else:
        peak_filtering_threshold_minutes = -10000
        time_minutes = 10
        strong_flare_threshold = "C5.0"  # inclusive to strong flares
        nan_removal_strategy = "linear_interpolation"
        scoring_metric = "precision"
        output_folder = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree"
        run_nickname = "newcm_test"
        model_type = "Tree"  # 'Tree', 'Random Forest' or 'Gradient Boosted Tree'
        multiclass = True  # else it's binary. If True, overrides strong_flare_threshold
        use_naive_diffs = True
        use_debug_mode = True
        grid_search(peak_filtering_threshold_minutes,
                    time_minutes,
                    strong_flare_threshold,
                    nan_removal_strategy,
                    scoring_metric,
                    use_naive_diffs,
                    output_folder,
                    run_nickname,
                    model_type,
                    multiclass,
                    use_debug_mode)

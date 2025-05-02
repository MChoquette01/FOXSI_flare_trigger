import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tree_common as tc

"""Analysis models using the ensemble test set"""


def get_temporal_test_results():
    """Returns a DataFrame of GT and predictions at each timestamp from the temporal test set"""

    # get trained trees
    trees = {}
    for minutes in range(-5, 16):
        trained_tree_path = os.path.join(results_folderpath, run_nickname, "Pruning", "Pruned Models", f"pruned_model_{minutes}_minutes_since_start.pkl")
        with open(trained_tree_path, "rb") as f:
            trees[minutes] = pickle.load(f)

    # using 0 minutes because the TEMPORAL test set is teh same for all timestamps
    split_datasets_path = os.path.join(results_folderpath, run_nickname, "Datasets", "split_datasets0_minutes_since_start.pkl")
    with open(split_datasets_path, 'rb') as f:
        split_datasets = pickle.load(f)

    temporal_test_gt_and_results = pd.concat([split_datasets["temporal_test"]["additional_data"].FlareID,
                                              split_datasets["temporal_test"]["additional_data"].FlareClass,
                                              split_datasets["temporal_test"]["additional_data"]["IsFlareClass>=C5"]], axis=1)
    columns = ["FlareID", "FlareClass", "IsFlareClass>=C5"]
    for minutes_since_start in range(-5, 16):
        trained_tree_path = os.path.join(results_folderpath, run_nickname, "Pruning", "Pruned Models", f"pruned_model_{minutes_since_start}_minutes_since_start.pkl")
        with open(trained_tree_path, "rb") as f:
            trained_tree = pickle.load(f)
        predictions = trained_tree.predict(split_datasets["temporal_test"]["x"])
        temporal_test_gt_and_results = pd.concat([temporal_test_gt_and_results, split_datasets["temporal_test"]["y"]], axis=1)
        temporal_test_gt_and_results = pd.concat([temporal_test_gt_and_results, pd.DataFrame(predictions)], axis=1)
        columns.append(f"GT{minutes_since_start}MinutesSinceStart")
        columns.append(f"Prediction{minutes_since_start}MinutesSinceStart")
    temporal_test_gt_and_results.columns = columns

    return temporal_test_gt_and_results


def make_cm_heatmaps(temporal_test_results):
    """Saves a heatmap of predictions versus guess to succinctly aggregate all confusion matricies"""

    flare_class_scores = {}
    for flare_class in ["B", "<C5", ">=C5", "M", "X"]:
        this_flare_class_scores = []
        xs = []
        ys = []
        for time_minutes in range(-5, 16):
            if not "C" in flare_class:
                subset = temporal_test_results[temporal_test_results.FlareClass.str.startswith(flare_class)]
            else:
                if flare_class == "<C5":
                    subset = temporal_test_results[(temporal_test_results.FlareClass.str.startswith("C")) & (temporal_test_results['IsFlareClass>=C5'] == False)]
                elif flare_class == ">=C5":
                    subset = temporal_test_results[(temporal_test_results.FlareClass.str.startswith("C")) & (temporal_test_results['IsFlareClass>=C5'] == True)]
            tn = subset[(subset[f"GT{time_minutes}MinutesSinceStart"] == 0) & (subset[f"Prediction{time_minutes}MinutesSinceStart"] == 0)].shape[0]
            fp = subset[(subset[f"GT{time_minutes}MinutesSinceStart"] == 0) & (subset[f"Prediction{time_minutes}MinutesSinceStart"] > 0)].shape[0]
            fn = subset[(subset[f"GT{time_minutes}MinutesSinceStart"] > 0) & (subset[f"Prediction{time_minutes}MinutesSinceStart"] == 0)].shape[0]
            tp = subset[(subset[f"GT{time_minutes}MinutesSinceStart"] > 0) & (subset[f"Prediction{time_minutes}MinutesSinceStart"] > 0)].shape[0]
            xs += [time_minutes] * (tn + fp + fn + tp)
            ys += [0] * tn
            ys += [1] * fp
            ys += [2] * fn
            ys += [3] * tp
            this_flare_class_scores.append({time_minutes: {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "xs": xs, "ys": ys}})
        flare_class_scores[flare_class] = this_flare_class_scores
        plt.figure(figsize=(16, 9))
        counts, xedges, yedges, im = plt.hist2d(xs, ys, bins=[21, 4], range=[[-5, 15], [0, 4]])
        cbar = plt.colorbar(im)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=18)
        y_step_size = (yedges[2] - yedges[1]) / 2
        tick_locs = [x - y_step_size for x in yedges[1:]]
        plt.yticks(ticks=tick_locs, labels=["TN", "FP", "FN", "TP"], fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlabel("Minutes Since Flare Start", fontsize=24)
        plt.ylabel("Prediction", fontsize=24)
        plt.title(f"Predictions for True Class: {flare_class}", fontsize=30)
        # plt.show()
        if "<" in flare_class:
            flare_class = flare_class.replace("<", "Less Than")
        elif ">=" in flare_class:
            flare_class = flare_class.replace(">=", "Greater Than")
        # plt.show()
        plt.savefig(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", "Heatmaps", f"Confusion Matrix Heatmap {flare_class} Class.png"))


def get_class_heatmap(temporal_test_results):
    """Saves a heatmap of predictions stratified by the true flare class"""

    for flare_class in ["B", "<C5", ">=C5", "M", "X"]:
        if not "C" in flare_class:
            subset = temporal_test_results[temporal_test_results.FlareClass.str.startswith(flare_class)]
        else:
            if flare_class == "<C5":
                subset = temporal_test_results[(temporal_test_results.FlareClass.str.startswith("C")) & (temporal_test_results['IsFlareClass>=C5'] == False)]
            elif flare_class == ">=C5":
                subset = temporal_test_results[(temporal_test_results.FlareClass.str.startswith("C")) & (temporal_test_results['IsFlareClass>=C5'] == True)]
        xs, ys = [], []
        for time_minutes in range(-5, 16):
            y = subset[f"Prediction{time_minutes}MinutesSinceStart"].values.tolist()
            for i, y in enumerate(y):
                xs.append(time_minutes)
                ys.append(y)
        plt.figure(figsize=(16, 9))
        counts, xedges, yedges, im = plt.hist2d(xs, ys, bins=[21, 4], range=[[-5, 15], [0, 4]])
        cbar = plt.colorbar(im)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=18)
        y_step_size = (yedges[2] - yedges[1]) / 2
        tick_locs = [x - y_step_size for x in yedges[1:]]
        plt.yticks(ticks=tick_locs, labels=["<C5", ">=C5", "M", "X"], fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlabel("Minutes Since Flare Start", fontsize=24)
        plt.ylabel("Prediction", fontsize=24)
        plt.title(f"Predictions for True Class: {flare_class}", fontsize=30)
        # plt.show()
        if "<" in flare_class:
            flare_class = flare_class.replace("<", "Less Than")
        elif ">=" in flare_class:
            flare_class = flare_class.replace(">=", "Greater Than")
        plt.savefig(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", "Heatmaps", f"Prediction History Heatmap {flare_class} Class.png"))
        # plt.show()


def consistency_analysis(temporal_test_results, formal_model_name):

    flare_accuracy_count = defaultdict(list)
    for minutes_since_start in range(-5, 16):
        gt_and_preds_at_time = temporal_test_results.loc[:, ["FlareID", f"GT{minutes_since_start}MinutesSinceStart", f"Prediction{minutes_since_start}MinutesSinceStart"]]
        for idx, row in gt_and_preds_at_time.iterrows():
            gt = bool(row[f"GT{minutes_since_start}MinutesSinceStart"])
            pred = bool(row[f"Prediction{minutes_since_start}MinutesSinceStart"])
            flare_accuracy_count[row.FlareID].append(gt == pred)

    # histogram of total accuracy counts
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    a = ax[0].hist([sum(x) for x in list(flare_accuracy_count.values())], bins=[x for x in range(22)])
    ax[0].set_xticks(ticks=[x for x in range(22)][::2], labels=[x for x in range(22)][::2], fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].set_ylabel("Frequency", fontsize=22)
    ax[0].set_xlabel("Number of Correct Predictions", fontsize=22)

    # Accuracy by minutes from start
    xs = []
    ys = []
    for minutes_since_start in range(-5, 16):
        minute_data = [x[minutes_since_start + 5] for x in list(flare_accuracy_count.values())]
        xs.append(minutes_since_start)
        ys.append(sum(minute_data) / len(minute_data))
        # minute_data = [x[minutes_since_start + 5] for x in list(flare_accuracy_count.values())]
    ax[1].plot(xs, ys)
    ax[1].set_xticks(ticks=[x for x in range(-5, 16)][::2], labels=[x for x in range(-5, 16)][::2], fontsize=18)
    ax[1].set_yticks(ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=18)
    ax[1].set_ylabel("Adjusted Accuracy", fontsize=22)
    ax[1].set_xlabel("Minutes Since Flare Start", fontsize=22)
    # ax[1].set_title(f"{formal_model_name}: Prediction Consistency", fontsize=24)
    # plt.show()
    plt.suptitle(f"{formal_model_name} Prediction Consistency", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", f"{formal_model_name} Temporal Test Set Accuracy By Minute.png"))

    # now, same thing for flares >= C5 GOES class
    strong_flare_subset = temporal_test_results[temporal_test_results["IsFlareClass>=C5"] == True]
    flare_accuracy_count = defaultdict(list)
    for minutes_since_start in range(-5, 16):
        gt_and_preds_at_time = strong_flare_subset.loc[:, ["FlareID", f"GT{minutes_since_start}MinutesSinceStart", f"Prediction{minutes_since_start}MinutesSinceStart"]]
        for idx, row in gt_and_preds_at_time.iterrows():
            gt = bool(row[f"GT{minutes_since_start}MinutesSinceStart"])
            pred = bool(row[f"Prediction{minutes_since_start}MinutesSinceStart"])
            flare_accuracy_count[row.FlareID].append(gt == pred)

    # histogram of total accuracy counts
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    a = ax[0].hist([sum(x) for x in list(flare_accuracy_count.values())], bins=[x for x in range(22)], align='mid')
    ax[0].set_xticks(ticks=[x for x in range(22)][::3], labels=[x for x in range(22)][::3], fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    # ax[0].set_yticks(fontsize=20)
    ax[0].set_ylabel("Frequency", fontsize=22)
    ax[0].set_xlabel("Number of Correct Predictions", fontsize=22)

    # Accuracy by minutes from start
    xs = []
    ys = []
    for minutes_since_start in range(-5, 16):
        minute_data = [x[minutes_since_start + 5] for x in list(flare_accuracy_count.values())]
        xs.append(minutes_since_start)
        ys.append(sum(minute_data) / len(minute_data))
        # minute_data = [x[minutes_since_start + 5] for x in list(flare_accuracy_count.values())]
    ax[1].plot(xs, ys)
    ax[1].set_xticks(ticks=[x for x in range(-5, 16)][::2], labels=[x for x in range(-5, 16)][::2], fontsize=18)
    ax[1].set_yticks(ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=18)
    ax[1].set_ylabel("Adjusted Accuracy", fontsize=22)
    ax[1].set_xlabel("Minutes Since Flare Start", fontsize=22)
    # plt.show()
    plt.suptitle(f"{formal_model_name} Prediction Consistency (GOES >= C5 Only)", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", f"{formal_model_name} Temporal Test Set Prediction Consistency Strong Flares2.png"))
    # now, same thing for flares >= C5 GOES class
    strong_flare_subset = temporal_test_results[temporal_test_results["IsFlareClass>=C5"] == True]
    flare_accuracy_count = defaultdict(list)
    for minutes_since_start in range(-5, 16):
        gt_and_preds_at_time = strong_flare_subset.loc[:, ["FlareID", f"GT{minutes_since_start}MinutesSinceStart",
                                                           f"Prediction{minutes_since_start}MinutesSinceStart"]]
        for idx, row in gt_and_preds_at_time.iterrows():
            gt = bool(row[f"GT{minutes_since_start}MinutesSinceStart"])
            pred = bool(row[f"Prediction{minutes_since_start}MinutesSinceStart"])
            flare_accuracy_count[row.FlareID].append(gt == pred)

if __name__ == "__main__":

    results_folderpath = r"MSI Results"
    run_nickname = r"2025_03_21_multiclass_naive_adjusted_precision_gbdt"
    formal_model_name = "Gradient Boosted Tree"

    # output folders
    if not os.path.exists(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", "Heatmaps")):
        os.makedirs(os.path.join(results_folderpath, run_nickname, "Temporal Analysis", "Heatmaps"))

    temporal_test_results = get_temporal_test_results()
    make_cm_heatmaps(temporal_test_results)
    get_class_heatmap(temporal_test_results)
    consistency_analysis(temporal_test_results, formal_model_name)
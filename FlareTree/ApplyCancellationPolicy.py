import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tree_common as tc


"""Functions to examine the effect of adding a launch cancellation policy, where if the XRSA flux three minutes after the timestep of
observation is lower, the launch is aborted."""


def get_original_and_cancelled_results(test_and_cancellation_results):
    """Returns GT< original predictions and predictions with cancellation criteria applied."""

    cancelled_y_true, cancelled_y_pred = [], []
    original_y_true, original_y_pred = [], []
    for idx, row in test_and_cancellation_results.iterrows():
        if int(float(row.GT)) == 0:
            if int(float(row.Prediction)) == 0 or int(float(row.IsCancelled)) == 1:
                cancelled_y_true.append(int(float(row.GT)))
                cancelled_y_pred.append(0)
            else:
                cancelled_y_true.append(int(float(row.GT)))
                cancelled_y_pred.append(int(float(row.Prediction)))
        else:
            if int(float(row.Prediction)) == 0 or int(float(row.IsCancelled)) == 1:
                cancelled_y_true.append(int(float(row.GT)))
                cancelled_y_pred.append(0)
            else:
                cancelled_y_true.append(int(float(row.GT)))
                cancelled_y_pred.append(int(float(row.Prediction)))
        original_y_true.append(int(float(row.GT)))
        original_y_pred.append(int(float(row.Prediction)))

    return original_y_true, original_y_pred, cancelled_y_pred


def graph_confusion_matrices(gt, old_preds, new_preds, run_nickname, time_minutes):
    """Helper function to plot confusion matricies with and without cancellation critieria applied"""

    fig, ax = plt.subplots(1, 2, clear=True)
    fig.set_figwidth(25)
    fig.set_figheight(14)
    ax[0].set_title("Without Cancellation", fontsize=24)
    ax[1].set_title("With Cancellation", fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=22)
    ax[1].tick_params(axis='both', which='major', labelsize=22)
    plt.rcParams.update({'font.size': 24})

    old_display_labels = [f"<C5", f"C, >=C5", f"M", f"X"][:len(list(set(gt + old_preds)))]
    new_display_labels = [f"<C5", f"C, >=C5", f"M", f"X"][:len(list(set(gt + new_preds)))]

    ConfusionMatrixDisplay.from_predictions(gt, old_preds, display_labels=old_display_labels).plot(ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(gt, new_preds, display_labels=new_display_labels).plot(ax=ax[1])

    ax[0].set_xlabel("Prediction", fontsize=24)
    ax[1].set_xlabel("Prediction", fontsize=24)
    ax[0].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)
    ax[1].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)

    if time_minutes < 0:
        fig.suptitle(f"Flare Start - {abs(time_minutes)} Minutes", fontsize=30)
    else:
        fig.suptitle(f"Flare Start + {abs(time_minutes)} Minutes", fontsize=30)
    plt.tight_layout()
    fig.savefig(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Cancellation Confusion Matrices", f"Confusion Matrix {time_minutes} Minutes Since Start.png"))
    # plt.show()
    plt.close(fig)
    plt.clf()
    plt.cla()


def plot_stratified_confusion_matricies(test_and_cancellation_results, time_minutes, run_nickname):
    """Helper function to plot confusion matricies stratified by true flare class"""

    for flare_class in ["B", "<C5", ">=C5", "M", "X"]:
        if not "C" in flare_class:
            relevant_indicies = test_and_cancellation_results.index[test_and_cancellation_results.FlareClass.str.startswith(flare_class)].tolist()
        else:
            if flare_class == "<C5":
                relevant_indicies = test_and_cancellation_results.index[(test_and_cancellation_results.FlareClass.str.startswith("C")) & (test_and_cancellation_results["IsFlareClass>=C5"] == "False")].tolist()
            elif flare_class == ">=C5":
                relevant_indicies = test_and_cancellation_results.index[(test_and_cancellation_results.FlareClass.str.startswith("C")) & (test_and_cancellation_results["IsFlareClass>=C5"] == "True")].tolist()

        subset = test_and_cancellation_results.iloc[relevant_indicies]
        gt, original_y_pred, cancelled_y_pred = get_original_and_cancelled_results(subset)

        train_display_labels = [f"< C5", f">= C5", f"M", f"X"][:len(list(set(gt + original_y_pred)))]
        test_display_labels = [f"< C5", f">= C5", f"M", f"X"][:len(list(set(gt + cancelled_y_pred)))]
        fig, ax = plt.subplots(1, 2, clear=True)
        fig.set_figwidth(25)
        fig.set_figheight(14)
        ax[0].set_title("Without Cancellation", fontsize=24)
        ax[1].set_title("With Cancellation", fontsize=24)
        ax[0].tick_params(axis='both', which='major', labelsize=22)
        ax[1].tick_params(axis='both', which='major', labelsize=22)
        plt.rcParams.update({'font.size': 26})

        ConfusionMatrixDisplay.from_predictions(gt, original_y_pred, display_labels=train_display_labels).plot(ax=ax[0])
        ConfusionMatrixDisplay.from_predictions(gt, cancelled_y_pred, display_labels=test_display_labels).plot(ax=ax[1])

        ax[0].set_xlabel("Prediction", fontsize=24)
        ax[1].set_xlabel("Prediction", fontsize=24)
        ax[0].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)
        ax[1].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)

        if time_minutes < 0:
            title_text = f"Flare Start - {abs(time_minutes)} Minutes"
        else:
            title_text = f"Flare Start + {abs(time_minutes)} Minutes"
        if "C" not in flare_class:
            title_text += f", True {flare_class} Class Flares Only"
        else:
            if "<" in flare_class:
                title_text += ", True C0.0 - C4.9 Class Flares Only"
            else:
                title_text += ", True C5.0 - C9.9 Class Flares Only"
        fig.suptitle(title_text, fontsize=30)
        plt.tight_layout()
        if "<" in flare_class:
            flare_class = flare_class.replace("<", "Less Than ")
        elif ">=" in flare_class:
            flare_class = flare_class.replace(">=", "Greater Than ")
        # fig.savefig(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Stratified Confusion Matrices", f"Confusion Matrix {time_minutes} Minutes Since Start True Class {flare_class}.png"))
        # plt.show()
        plt.close(fig)
        plt.clf()
        plt.cla()

def is_flare_cancelled(flare_id, minutes_since_start, current_xrsa):

    if current_xrsa is None:
        return False
    client, flares_table = tc.connect_to_flares_db()
    minutes_out = 3
    complete = False
    while not complete:
        cursor = flares_table.find(filter={'FlareID': f'{int(flare_id)}_{minutes_since_start + 15 + minutes_out}'}, projection={'CurrentXRSA': 1})
        for record in cursor:
            if record["CurrentXRSA"] is not None:
                newer_xrsa = record["CurrentXRSA"]
                if newer_xrsa < current_xrsa:
                    client.close()
                    return True
                complete = True
            else:  # try a timestep earlier, this is what would be used by linear interpolation
                minutes_out -= 1
                if minutes_out == 0:  # no point going earlier than the current timestep
                    client.close()
                    return False
        else:  # no data? shouldn't happen
            client.close()
            return False


def make_cancelled_launch_graph_for_positive_predictions():

    all_results = {}
    for x in range(-5, 16):
        all_results[x] = 0
    positive_prediction_flare_ids = {}
    for minutes_since_start in range(-5, 16):
        test_set_results_filepath = os.path.join(results_folderpath, run_nickname, "Scores", f"{minutes_since_start}_minutes_since_start_test_set.csv")
        test_set_results = pd.read_csv(test_set_results_filepath)
        positive_test_set_predictions = test_set_results[test_set_results.Prediction >= 1]
        positive_prediction_test_ids = [row.FlareID for _, row in positive_test_set_predictions.iterrows()]
        positive_prediction_flare_ids[minutes_since_start] = positive_prediction_test_ids

    client, flares_table = tc.connect_to_flares_db()
    for minutes_since_start in range(-5, 16):
        for flare_id in positive_prediction_flare_ids[minutes_since_start]:
            cursor = client['Flares']['NaiveFlares'].find(filter={'FlareID': f'{int(flare_id)}_{minutes_since_start + 15}'}, projection={'CurrentXRSA': 1})
            for record in cursor:
                current_xrsa = record["CurrentXRSA"]
                if is_flare_cancelled(flare_id, minutes_since_start, current_xrsa):
                    all_results[minutes_since_start] += 1
    client.close()
    all_results = dict(sorted(all_results.items()))
    all_results = [all_results[minutes_since_start] / len(positive_prediction_flare_ids[minutes_since_start]) for minutes_since_start in range(-5, 16)]
    plt.figure(figsize=(16, 9))
    plt.bar(range(-5, 16), all_results)
    plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Proportion of Positive Predictions Cancelled", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=22)
    plt.title("Flare Cancellations (Positive Predictions Only)", fontsize=24)
    # plt.show()
    # plt.savefig(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Cancelled Flares Positive Predictions.png"))


def analyze_cancellation_changes():
    """Make confusion matricies of before/after cancellation effects applied, for the entire model and stratiified by
    true flare class. Saves images and a CSV of scoring differences."""

    stats = []
    client, flares_table = tc.connect_to_flares_db()
    for time_minutes in tqdm(range(-5, 16), desc="Analyzing effects of cancellation..."):
        test_set_results_filepath = os.path.join(results_folderpath, run_nickname, "Pruning", "Pruned Scores", f"{time_minutes}_minutes_since_start_test_set.pkl")
        split_datasets_filepath = os.path.join(results_folderpath, run_nickname, "Datasets", f"split_datasets{time_minutes}_minutes_since_start.pkl")
        with open(test_set_results_filepath, "rb") as f:
            test_set_results = pickle.load(f)
        with open(split_datasets_filepath, "rb") as f:
            split_datasets = pickle.load(f)
        launch_cancelled_status = []
        all_additional_data = pd.concat([split_datasets["test"]["additional_data"], split_datasets["test"]["y"]], axis=1)
        for idx, row in all_additional_data.iterrows():
            cursor = client['Flares']['NaiveFlares'].find(filter={'FlareID': f'{int(row.FlareID)}_{time_minutes + 15}'}, projection={'CurrentXRSA': 1})
            for record in cursor:
                if record["CurrentXRSA"] is not None:
                    current_xrsa = record["CurrentXRSA"]
            if is_flare_cancelled(row.FlareID, time_minutes, current_xrsa):
                launch_cancelled_status.append(1)
            else:
                launch_cancelled_status.append(0)
        test_and_cancellation_results = pd.concat([test_set_results, pd.DataFrame(np.array(launch_cancelled_status))], axis=1)
        test_and_cancellation_results.columns = list(test_set_results.columns) + ["IsCancelled"]

        gt, original_y_pred, cancelled_y_pred = get_original_and_cancelled_results(test_and_cancellation_results)

        old_cm = confusion_matrix(gt, original_y_pred)
        new_cm = confusion_matrix(gt, cancelled_y_pred)
        old_adj_precision = old_cm[1:4, 1:4].sum() / (old_cm[1:4, 1:4].sum() + sum(old_cm[0, 1:4]))
        old_adj_recall = old_cm[1:4, 1:4].sum() / (old_cm[1:4, 1:4].sum() + sum(old_cm[1:4, 0]))
        old_adj_f1 = (2 * old_adj_precision * old_adj_recall) / (old_adj_precision + old_adj_recall)
        new_adj_precision = new_cm[1:4, 1:4].sum() / (new_cm[1:4, 1:4].sum() + sum(new_cm[0, 1:4]))
        new_adj_recall = new_cm[1:4, 1:4].sum() / (new_cm[1:4, 1:4].sum() + sum(new_cm[1:4, 0]))
        new_adj_f1 = (2 * new_adj_precision * new_adj_recall) / (new_adj_precision + new_adj_recall)
        stats.append([time_minutes, old_adj_precision, new_adj_precision, new_adj_precision - old_adj_precision,
                      old_adj_recall, new_adj_recall, new_adj_recall - old_adj_recall,
                      old_adj_f1, new_adj_f1, new_adj_f1 - old_adj_f1])

        plot_stratified_confusion_matricies(test_and_cancellation_results, time_minutes, run_nickname)
        graph_confusion_matrices(gt, original_y_pred, cancelled_y_pred, run_nickname, time_minutes)

    stats = pd.DataFrame(np.array(stats))
    stats.columns = ["MinutesSinceFlareStart", "AdjustedPrecisionWithoutCancellation", "AdjustedPrecisionWithCancellation",
                     "AdjustedPrecisionDifference", "AdjustedRecallWithoutCancellation", "AdjustedRecallWithCancellation",
                     "AdjustedRecallDifference", "AdjustedF1WithoutCancellation", "AdjustedF1WithCancellation", "AdjustedF1Difference"]
    stats.to_csv(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Cancellation Confusion Matrices", "Cancellation Effects.csv"), index=False)



if __name__ == "__main__":

    results_folderpath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results"
    run_nickname = r"2025_03_06_multiclass_adjusted_precision_gbdt_debug"

    # output folders
    if not os.path.exists(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Stratified Confusion Matrices")):
        os.makedirs(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Stratified Confusion Matrices"))
    if not os.path.exists(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Cancellation Confusion Matrices")):
        os.makedirs(os.path.join(results_folderpath, run_nickname, "Cancellation Analysis", "Cancellation Confusion Matrices"))

    make_cancelled_launch_graph_for_positive_predictions()
    analyze_cancellation_changes()


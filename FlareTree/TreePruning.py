import tree_common as tc
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree._tree import TREE_LEAF
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import numpy as np


def make_confusion_matrix(out_dir, t, train_x, train_y, test_x, test_y, minutes_since_start, max_depth=None, ccp_alpha=None):
    """Create training and test confusion matrices"""

    test_predictions = t.predict(test_x)
    train_predictions = t.predict(train_x)

    plt.clf()
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(16)
    fig.set_figheight(9)
    ax[0].set_title("Training", fontsize=14)
    ax[1].set_title("Test", fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[0].set_xlabel(["Predicted Label"], fontsize=14)
    ax[1].set_xlabel(["Predicted Label"], fontsize=14)
    ax[0].set_ylabel(["True Label"], fontsize=14)
    ax[1].set_ylabel(["True Label"], fontsize=14)
    plt.rcParams.update({'font.size': 16})
    ConfusionMatrixDisplay.from_predictions(train_y, train_predictions, display_labels=["< C5", ">= C5"]).plot(ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(test_y, test_predictions, display_labels=["< C5", ">= C5"]).plot(ax=ax[1])
    if max_depth is not None:
        fig.suptitle(f"Flares {minutes_since_start - 15} minutes since start, Max Depth: {max_depth}")
    elif ccp_alpha is not None:
        fig.suptitle(f"Flares {minutes_since_start - 15} minutes since start, CCP Alpha: {ccp_alpha}")
    else:
        fig.suptitle(f"Flares {minutes_since_start - 15} minutes since start")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ConfusionMatrices_{minutes_since_start}_minutes_since_start.png"))
    # plt.show()


def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def prune_ccp_alphas(peak_filtering_minutes):

    out_dir = os.path.join(os.path.split(results_filepath)[0], "Pruning", "CCP Alpha")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # ccp pruning
    for timestamp in results.minutes_since_start.tolist():
        # get data
        train_x, _, train_y, test_x, _, test_y = tc.get_train_and_test_data_from_pkl(int(timestamp), peak_filtering_minutes=peak_filtering_minutes)
        train_x, test_x = tc.impute_variable_data(train_x, test_x)

        # get baseline
        t = tc.create_tree_from_df(results, int(timestamp))
        t.fit(train_x, train_y)
        train_predictions = t.predict(train_x.values)
        train_y_reshaped = train_y.to_numpy().reshape((np.shape(train_y)[0],))
        train_cm = confusion_matrix(train_y, train_predictions)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_f1 = f1_score(train_y, train_predictions)
        make_confusion_matrix(out_dir, t, train_x, train_y, test_x, test_y, int(timestamp))

        best_f1 = train_f1
        best_alpha = None

        path = t.cost_complexity_pruning_path(train_x, train_y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        for alpha in ccp_alphas:
            if alpha < 0:
                continue
            t_pruned = tc.create_tree_from_df(results, int(timestamp), ccp_alpha=alpha)
            t_pruned.fit(train_x, train_y)
            pruned_test_predictions = t_pruned.predict(test_x.values)
            pruned_train_predictions = t_pruned.predict(train_x.values)
            pruned_test_cm = confusion_matrix(test_y, pruned_test_predictions)
            pruned_test_tn, pruned_test_fp, pruned_test_fn, pruned_test_tp = pruned_test_cm.ravel()
            pruned_train_f1 = balanced_accuracy_score(train_y, pruned_train_predictions)
            if pruned_test_tn != 0 and pruned_test_fp != 0 and pruned_test_fn != 0 and pruned_test_tp != 0:
                if pruned_train_f1 > best_f1:
                    best_f1 = pruned_train_f1
                    best_alpha = alpha
        print(f"Timestamp: {timestamp}: Best alpha: {best_alpha} with an F1 of {best_f1}, an increase of {best_f1 - train_f1}")
        make_confusion_matrix(out_dir, t_pruned, train_x, train_y, test_x, test_y, int(timestamp), ccp_alpha=best_alpha)


def prune_max_depth(peak_filtering_minutes):

    out_dir = os.path.join(os.path.split(results_filepath)[0], "Pruning", "Max Depth")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for timestamp in results.minutes_since_start.tolist():
        # get data
        train_x, _, train_y, test_x, _, test_y = tc.get_train_and_test_data_from_pkl(int(timestamp), peak_filtering_minutes=peak_filtering_minutes)
        train_x, test_x = tc.impute_variable_data(train_x, test_x)

        # get baseline
        t = tc.create_tree_from_df(results, int(timestamp))
        t.fit(train_x, train_y)
        tree_depth = t.tree_.max_depth
        train_predictions = t.predict(train_x.values)
        train_y_reshaped = train_y.to_numpy().reshape((np.shape(train_y)[0],))
        train_cm = confusion_matrix(train_y, train_predictions)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_f1 = f1_score(train_y, train_predictions)
        # make_confusion_matrix(t, train_x, train_y, test_x, test_y, int(timestamp))

        best_f1 = train_f1
        best_max_depth = None

        fewer_layers = list(range(1, tree_depth, 1))
        for layer in fewer_layers:
            max_depth_for_timestamp = results[results.minutes_since_start == timestamp].max_depth
            t_pruned = tc.create_tree_from_df(results, int(timestamp), max_depth_override=int(max_depth_for_timestamp.iloc[0]) - layer)
            t_pruned.fit(train_x, train_y)
            pruned_test_predictions = t_pruned.predict(test_x.values)
            pruned_train_predictions = t_pruned.predict(train_x.values)
            pruned_test_cm = confusion_matrix(test_y, pruned_test_predictions)
            pruned_test_tn, pruned_test_fp, pruned_test_fn, pruned_test_tp = pruned_test_cm.ravel()
            pruned_train_f1 = balanced_accuracy_score(train_y, pruned_train_predictions)
            if pruned_test_tn != 0 and pruned_test_fp != 0 and pruned_test_fn != 0 and pruned_test_tp != 0:
                if pruned_train_f1 > best_f1:
                    best_f1 = pruned_train_f1
                    best_max_depth = int(results.max_depth.iloc[0]) - layer
        print(f"Timestamp: {timestamp}: Best max depth: {best_max_depth} with an F1 of {best_f1}, an increase of {best_f1 - train_f1}")
        make_confusion_matrix(out_dir, t_pruned, train_x, train_y, test_x, test_y, int(timestamp), max_depth=best_max_depth)


if __name__ == "__main__":

    results_folderpath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results"
    run_nickname = "F1_filter_past_peak_flares"
    peak_filtering_minutes = 0

    out_dir = os.path.join(results_folderpath, "Pruning")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    results = tc.get_results_pickle(results_folderpath, run_nickname)

    prune_max_depth(peak_filtering_minutes)
    prune_ccp_alphas(peak_filtering_minutes)
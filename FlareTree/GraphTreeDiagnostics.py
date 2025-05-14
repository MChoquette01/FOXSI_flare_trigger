import matplotlib.pyplot as plt
import pickle
import math
import tree_common as tc
from sklearn.metrics import ConfusionMatrixDisplay
import os
from collections import defaultdict

"""Plots to gauge and debug tree performance"""


def metric_plot_helper(plot_int, train_list, test_list, metric_name):
    """Helper for make_metric_plot()"""

    xs = [int(x) - 15 for x in results.minutes_since_start.to_list()]

    plt.subplot(plot_int)
    train_metric = [float(x) for x in train_list]
    test_metric = [float(x) for x in test_list]
    for x, train_sample, test_sample in zip(xs, train_metric, test_metric):
        plt.plot([x, x], [train_sample, test_sample], color="black")
    plt.scatter(xs, train_metric, color="blue", label="Train")
    plt.scatter(xs, test_metric, color="orange", label="Test")
    if metric_name != "TSS":  # different range
        plt.yticks(ticks=[0, 1], labels=["0", "1"])
    else:
        plt.yticks(ticks=[-1, 0, 1], labels=["-1", "0", "1"])
    plt.xlabel("Minutes Since Start")
    plt.title(metric_name)
    plt.legend()


def param_plot_helper(plot_int, metric_results, metric_name):
    """Helper for make_param_plot()"""

    xs = [int(x) - 15 for x in results.minutes_since_start.to_list()]
    plt.subplot(plot_int)
    if metric_name != "Criterion":
        plt.scatter(xs, [float(x) for x in metric_results], color="blue")
    else:
        splits = {"gini": {"xs": [], "ys": []}, "entropy": {"xs": [], "ys": []}, "log_loss": {"xs": [], "ys": []}}
        for x, metric_result in zip(xs, metric_results):
            splits[metric_result]["xs"].append(x)
            splits[metric_result]["ys"].append(1)
        plt.scatter(splits["gini"]["xs"], splits["gini"]["ys"], color="blue", label="Gini")
        plt.scatter(splits["entropy"]["xs"], splits["entropy"]["ys"], color="red", label="Entropy")
        plt.scatter(splits["log_loss"]["xs"], splits["log_loss"]["ys"], color="green", label="Log Loss")
        plt.yticks(ticks=[0, 1, 2], labels=["0", "1", "2"])
        plt.legend()
    plt.xlabel("Minutes Since Start")
    plt.title(metric_name)


def make_metric_plot():
    """Plot model performance metrics (precision, recall, F1, etc.)"""

    plt.figure(figsize=(16, 9))
    metric_plot_helper(231, results.train_precision.to_list(), results.test_precision.to_list(), "Precision")
    metric_plot_helper(232, results.train_recall.to_list(), results.test_recall.to_list(), "Recall")
    metric_plot_helper(233, results.train_f1.to_list(), results.test_f1.to_list(), "F1")
    metric_plot_helper(234, results.train_tpr.to_list(), results.test_tpr.to_list(), "TPR")
    metric_plot_helper(235, results.train_fpr.to_list(), results.test_fpr.to_list(), "FPR")
    metric_plot_helper(236, results.train_tss.to_list(), results.test_tss.to_list(), "TSS")
    plt.suptitle("Optimal Tree Performance from GridSearchCV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"TreePerformanceMetrics.png"))
    # plt.show()


def make_param_plot():
    """Plot the best hyperparameters found by GridSearchCV for each timestamp"""

    plt.figure(figsize=(16, 9))
    param_plot_helper(231, results.criterion.to_list(), "Criterion")
    param_plot_helper(232, results.max_depth.to_list(), "Max Depth")
    param_plot_helper(233, results.min_samples_split.to_list(), "Min Samples Split")
    param_plot_helper(234, results.min_samples_leaf.to_list(), "Min Samples Leaf")
    param_plot_helper(235, results.min_weight_fraction_leaf.to_list(), "Min Weight Fraction Leaf")
    param_plot_helper(236, results.max_features.to_list(), "Max Features")
    plt.suptitle("Tree Hyperparameters from GridSearchCV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"TreeHyperparameterPlot.png"))
    # plt.show()


def graph_nan_frequency(results, peak_filtering_minutes):
    """Plot the proportion of NaN values for each timestamp"""

    plt.clf()
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(16)
    fig.set_figheight(9)
    metric_values_train = defaultdict(list)
    metric_values_test = defaultdict(list)
    for timestamp in results.minutes_since_start.to_list():
        split_datasets = tc.get_train_and_test_data_from_pkl(int(timestamp), peak_filtering_minutes=peak_filtering_minutes, stratify=stratify, use_science_delay=use_science_delay)
        training_record_count = split_datasets["train"]["x"].shape[0]
        test_record_count = split_datasets["test"]["x"].shape[0]
        for metric in ["Temperature", "Temperature1MinuteDifference", "Temperature3MinuteDifference",
                       "Temperature5MinuteDifference", "EmissionMeasure", "EmissionMeasure1MinuteDifference",
                       "EmissionMeasure3MinuteDifference", "EmissionMeasure5MinuteDifference", "CurrentXRSA", "CurrentXRSB"]:
            metric_values_train[metric].append(split_datasets["train"]["x"][metric].isna().sum() / training_record_count)
            metric_values_test[metric].append(split_datasets["test"]["x"][metric].isna().sum() / test_record_count)
    # plot
    xs = [int(x) for x in results.minutes_since_start.to_list()]
    for metric, values in metric_values_train.items():
        ax[0].plot([x - 15 for x in xs], values, label=metric)
    for metric, values in metric_values_test.items():
        ax[1].plot([x - 15 for x in xs], values, label=metric)
    ax[0].legend()
    ax[0].set_title("Training", fontsize=22)
    ax[1].set_title("Test", fontsize=22)
    ax[0].set_xlabel("Minutes Since Flare Start", fontsize=14)
    # ax[0].set_xticks(ticks=xs, labels=xs, fontsize=14)
    ax[0].set_ylabel("Percent of NaN Values", fontsize=14)
    plt.suptitle("NaN Frequency")
    # ax[0].set_yticks(ticks=[x / 10 for x in range(0, 11, 1)], labels=[x / 10 for x in range(0, 11, 1)], fontsize=14)
    plt.savefig(os.path.join(out_dir, "nan_frequency.png"))
    # plt.show()


def make_confusion_matrix(results, minutes_since_start, peak_filtering_minutes):
    """Create training and test confusion matrices"""

    t = tc.create_tree_from_df(results, minutes_since_start)
    split_datasets = tc.get_train_and_test_data_from_pkl(minutes_since_start, peak_filtering_minutes=peak_filtering_minutes, stratify=stratify, use_science_delay=use_science_delay)
    t.fit(split_datasets["train"]["x"], split_datasets["train"]["y"])
    test_predictions = t.predict(split_datasets["test"]["x"])
    train_predictions = t.predict(split_datasets["train"]["x"])

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
    ConfusionMatrixDisplay.from_predictions(split_datasets["train"]["y"], train_predictions, display_labels=["< C5", ">= C5"]).plot(ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(split_datasets["test"]["y"], test_predictions, display_labels=["< C5", ">= C5"]).plot(ax=ax[1])
    fig.suptitle(f"Flares {minutes_since_start - 15} minutes since start", fontsize=24)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ConfusionMatrices_{minutes_since_start}_minutes_since_start.png"))
    # plt.show()


def graph_feature_importance(minutes_since_start, peak_filtering_minutes):
    """Create a bar chart showing tree feature importance"""

    t = tc.create_tree_from_df(results, minutes_since_start)
    split_datasets = tc.get_train_and_test_data_from_pkl(minutes_since_start, peak_filtering_minutes=peak_filtering_minutes, stratify=stratify, use_science_delay=use_science_delay)
    t.fit(split_datasets["train"]["x"], split_datasets["train"]["y"])
    features_importances = t.feature_importances_
    f_i = []
    nans = []
    for idx, feature_importance in enumerate(features_importances):
        if not math.isnan(feature_importance):
            f_i.append([split_datasets["train"]["x"].columns[idx], feature_importance])
        else:
            nans.append([split_datasets["train"]["x"].columns[idx], 0.1])
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
    plt.savefig(os.path.join(out_dir, f"FeatureImportance_{minutes_since_start}_minutes_since_start.png"))
    # plt.show()


def graph_flare_count(results, strong_flare_threshold):
    """Graph the flare count over all timestamps"""

    xs = [int(x) for x in results.minutes_since_start.to_list()]
    weak_flare_count = [int(x) for x in results["number_of_strong_flares"]]
    strong_flare_count = [int(x) for x in results["number_of_strong_flares"]]
    total_number_of_flares = [int(x) for x in results.total_number_of_flares]
    plt.figure(figsize=(16, 9))
    plt.plot(xs, weak_flare_count, color="red", label=f"<{strong_flare_threshold} Count")
    plt.plot(xs, strong_flare_count, color="green", label=f">={strong_flare_threshold} Count")
    plt.plot(xs, total_number_of_flares, color="black", label="Total")
    plt.xlabel("Minutes Since Start")
    plt.title("Count of Flares")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "FlareCount.png"))
    # plt.show()


results_folderpath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results"
run_nickname = "2025_03_08_multiclass_naive_adjusted_precision_rf"
inputs = tc.get_inputs_dict(results_folderpath, run_nickname)
peak_filtering_minutes = inputs['peak_filtering_threshold_minutes']
strong_flare_threshold = inputs['strong_flare_threshold']
use_naive_diffs = inputs['use_naive_diffs']
# use_science_delay = inputs["use_science_delay"]
stratify = inputs["stratify"]

results = tc.get_results_pickle(results_folderpath, run_nickname)

out_dir = os.path.join(results_folderpath, run_nickname, "Analysis")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

make_param_plot()
make_metric_plot()
graph_flare_count(results, strong_flare_threshold)
graph_nan_frequency(results, peak_filtering_minutes)
# for timestamp in results.minutes_since_start.tolist():
#     make_confusion_matrix(results, minutes_since_start=int(timestamp), peak_filtering_minutes=peak_filtering_minutes)
#     graph_feature_importance(minutes_since_start=int(timestamp), peak_filtering_minutes=peak_filtering_minutes)

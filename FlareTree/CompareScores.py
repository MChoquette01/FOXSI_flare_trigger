import matplotlib.pyplot as plt
import os
import pickle

# Script to compare metrics from two decision tree grid searches


def metric_plot_helper(plot_int, old_list, new_list, metric_name):
    """Helper for make_metric_plot()"""

    xs = [int(x) - 15 for x in old_results.minutes_since_start.to_list()]

    plt.subplot(plot_int)
    old_metric = [float(x) for x in old_list]
    new_metric = [float(x) for x in new_list]
    for x, train_sample, test_sample in zip(xs, old_metric, new_metric):
        plt.plot([x, x], [train_sample, test_sample], color="black")
    plt.scatter(xs, old_metric, color="blue", label=old_version_nickname)
    plt.scatter(xs, new_metric, color="orange", label=new_version_nickname)
    if metric_name != "TSS":
        plt.yticks(ticks=[0, 1], labels=["0", "1"])
    else:
        plt.yticks(ticks=[-1, 0, 1], labels=["-1", "0", "1"])
    plt.xlabel("Minutes Since Start")
    plt.title(metric_name)
    plt.legend()


def make_metric_plot(old_results, new_results):
    """Plot model performance metrics"""

    plt.figure(figsize=(16, 9))
    metric_plot_helper(231, old_results.test_precision.to_list(), new_results.test_precision.to_list(), "Precision")
    metric_plot_helper(232, old_results.test_recall.to_list(), new_results.test_recall.to_list(), "Recall")
    metric_plot_helper(233, old_results.test_f1.to_list(), new_results.test_f1.to_list(), "F1")
    metric_plot_helper(234, old_results.test_fpr.to_list(), new_results.test_fpr.to_list(), "FPR")
    metric_plot_helper(235, old_results.test_tss.to_list(), new_results.test_tss.to_list(), "TSS")
    plt.suptitle("Optimal Tree Performance from GridSearchCV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{old_version_nickname}.png"))
    # plt.show()


root_dir = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results"
old_version_nickname = "F1_filter_past_peak_flares"
new_version_nickname = "F1_no_peak_filtering"

old_results_filepath = os.path.join(root_dir, old_version_nickname, "results.pkl")
new_results_filepath = os.path.join(root_dir, new_version_nickname, "results.pkl")

with open(old_results_filepath, "rb") as f:
    old_results = pickle.load(f)

with open(new_results_filepath, "rb") as f:
    new_results = pickle.load(f)

out_dir = os.path.join(os.path.split(new_results_filepath)[0], "Comparison")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

make_metric_plot(old_results, new_results)
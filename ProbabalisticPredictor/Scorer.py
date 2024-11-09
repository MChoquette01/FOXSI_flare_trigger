import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

"""Some scoring tools for the output of BatchRunThresholds.py"""

linear_data_filepath = "Linear Bin_df_sql.pkl"
log_data_filepath = "log2_df.pkl"
percent_data_filepath = "percent_df.pkl"

with open(linear_data_filepath, "rb") as f:
    linear_df = pickle.load(f)


def get_true_class(flare_id):

    true_class = linear_df[(linear_df.FlareID == flare_id)]
    true_class.sort_values(["FlareTimestamp"])
    return true_class.BaseFlareClass.iloc[0]


def calculate_tss(cm_array):

    fp_total, fn_total, tp_total, tn_total = 0, 0, 0, 0
    for dim in range(cm_array.shape[0]):
        cm = cm_array[dim]
        tn, fp, fn, tp = cm.ravel()
        fp_total += fp
        fn_total += fn
        tp_total += tp
        tn_total += tn
    fpr = tn_total / (tn_total + fp_total)
    tpr = tp_total / (tp_total + fn_total)
    return tpr, fpr, tpr - fpr


# fp.append(cm.sum(axis=0) - np.diag(cm))
# fn.append(cm.sum(axis=1) - np.diag(cm))
# tp.append(np.diag(cm))
# tn.append(cm.sum() - (fp + fn + tp))


# get all unique flare IDs
first_flares = linear_df.FlareID.unique().tolist()

# get unique param combinations
unique_bin_combinations = linear_df.groupby("XRSBFluxBinSize").XRSB3MinuteDifferenceBinSize.unique()

benchmark = 0.8
minimum_flare_count = 5
CLASS_LABELS = ["B", "C", "M", "X"]


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

def unweighted_confusion_matrix():
    """Treat each similar flare as a datapoint, add them together (unweighted) and make a confusion matrix"""

    results = {}
    for xrsb_bin_value, xrsb_difference_bin_values in unique_bin_combinations.items():
        for xrsb_difference_bin_value in xrsb_difference_bin_values:
            threshold_results = {}
            flare_results = {}
            gt_tally = []
            pred_tally = []
            for flare_id in first_flares:
                class_gt = get_true_class(flare_id)
                # get flare class predictions
                class_predictions = linear_df[(linear_df.XRSBFluxBinSize == xrsb_bin_value) &
                                              (linear_df.XRSB3MinuteDifferenceBinSize == xrsb_difference_bin_value) &
                                              (linear_df.FlareID == flare_id)]
                unique_timestamps = class_predictions.FlareTimestamp.unique()
                timestamp_results_ret = []
                for timestamp in unique_timestamps:
                    this_timestamp_results_ret = {}
                    timestamp_results = class_predictions[class_predictions.FlareTimestamp == timestamp]
                    total_flare_count = int(timestamp_results.ACount.iloc[0]) + int(timestamp_results.BCount.iloc[0]) + int(timestamp_results['C<5Count'].iloc[0]) +\
                                        int(timestamp_results['C>=5Count'].iloc[0]) + int(timestamp_results.MCount.iloc[0]) + int(timestamp_results.XCount.iloc[0]) +\
                                        int(timestamp_results.UnknownClass.iloc[0])
                    if class_gt[0] != "C":
                        gt_count = timestamp_results[f"{class_gt[0]}Count"]
                    else:
                        if float(class_gt[1:]) < 5.0:
                            gt_count = timestamp_results["C<5Count"]
                        else:
                            gt_count = timestamp_results["C>=5Count"]
                    if total_flare_count != 0:
                        percent_true = int(gt_count.iloc[0]) / int(total_flare_count)
                    else:
                        percent_true = 0  # TODO: maybe should be nan or inf
                    flare_class_columns = timestamp_results.iloc[:, 6:11]
                    max_flare = str(flare_class_columns.idxmax(axis=1).values)[2]
                    pred_tally.append(max_flare)
                    gt_tally.append(class_gt[0])
                    this_timestamp_results_ret[timestamp] = {"PercentTrue": percent_true,
                                                    "TotalFlareCount": total_flare_count,
                                                    "MeetsMinimumFlareCount": True if total_flare_count >= minimum_flare_count else False,
                                                    "MeetsBenchmarkAccuracy": True if percent_true >= benchmark else False}
                    timestamp_results_ret.append(this_timestamp_results_ret)
                flare_results[flare_id] = timestamp_results_ret
            # threshold_results[('{:.2E}'.format(float(xrsb_bin_value)), '{:.2E}'.format(float(xrsb_difference_bin_value)))] = flare_results
            # results.append(threshold_results)
            cm_layers = multilabel_confusion_matrix(y_true=gt_tally, y_pred=pred_tally, labels=CLASS_LABELS)
            tpr, fpr, tss = calculate_tss(cm_layers)
            ConfusionMatrixDisplay.from_predictions(y_true=gt_tally, y_pred=pred_tally, labels=CLASS_LABELS)
            plt.title(f"XRSB Value: {'{:.2E}'.format(float(xrsb_bin_value))}, XRSB 3 Min Diff Value: {'{:.2E}'.format(float(xrsb_difference_bin_value))}")
            plt.show()
            this_result = {"precision": precision_score(y_true=gt_tally, y_pred=pred_tally, labels=CLASS_LABELS, average="micro"),
                           "recall": recall_score(y_true=gt_tally, y_pred=pred_tally, labels=CLASS_LABELS, average="micro"),
                           "f1": f1_score(y_true=gt_tally, y_pred=pred_tally, labels=CLASS_LABELS, average="micro"),
                           "tss": tss,
                           "tpr": tpr,
                           "fpr": fpr}
            results[('{:.2E}'.format(float(xrsb_bin_value)), '{:.2E}'.format(float(xrsb_difference_bin_value)))] = this_result

    plt.clf()
    f1_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["f1"], reverse=True)}
    precision_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["precision"], reverse=True)}
    recall_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["recall"], reverse=True)}
    tss_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["tss"], reverse=True)}
    tpr_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["tpr"], reverse=True)}
    fpr_sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]["fpr"], reverse=True)}
    xs = range(len(list(f1_sorted_results.keys())))
    plt.bar([x - 0.25 for x in xs], [x["precision"] for x in tss_sorted_results.values()], width=0.1, label="Precision")
    plt.bar([x - 0.15 for x in xs], [x["recall"] for x in tss_sorted_results.values()], width=0.1, label="Recall")
    plt.bar([x - 0.05 for x in xs], [x["f1"] for x in tss_sorted_results.values()], width=0.1, label="F1")
    plt.bar([x + 0.05 for x in xs], [x["fpr"] for x in tss_sorted_results.values()], width=0.1, label="FPR")
    plt.bar([x + 0.15 for x in xs], [x["tpr"] for x in tss_sorted_results.values()], width=0.1, label="TPR")
    plt.bar([x + 0.25 for x in xs], [x["tss"] for x in tss_sorted_results.values()], width=0.1, label="TSS")
    plt.xticks(xs, list(f1_sorted_results.keys()), rotation=90)
    plt.tight_layout()
    plt.xlabel("Thresholds (Current XRSB, 3 Min XRSB Diff)")
    plt.ylim((-1, 1))
    plt.legend()
    plt.show()


def find_c5_threshold():
    """Lots of B flares...no historical predictor will find 50% of similar flares as C consistently.
    Find a percentage level of which similar flares are >= C5 that demonstrates confidence."""

    results = []
    for xrsb_bin_value, xrsb_difference_bin_values in unique_bin_combinations.items():
        for xrsb_difference_bin_value in xrsb_difference_bin_values:
            threshold_results = {}
            flare_results = {}
            for flare_id in first_flares:
                class_gt = get_true_class(flare_id)
                # get flare class predictions
                class_predictions = linear_df[(linear_df.XRSBFluxBinSize == xrsb_bin_value) &
                                              (linear_df.XRSB3MinuteDifferenceBinSize == xrsb_difference_bin_value) &
                                              (linear_df.FlareID == flare_id)]
                unique_timestamps = class_predictions.FlareTimestamp.unique()
                timestamp_results_ret = []
                for timestamp in unique_timestamps:
                    this_timestamp_results_ret = []
                    timestamp_results = class_predictions[class_predictions.FlareTimestamp == timestamp]
                    under_c5_flare_count = int(timestamp_results.ACount.iloc[0]) + int(timestamp_results.BCount.iloc[0]) + int(timestamp_results['C<5Count'].iloc[0])
                    c5_and_over_count = int(timestamp_results['C>=5Count'].iloc[0]) + int(timestamp_results.MCount.iloc[0]) + int(timestamp_results.XCount.iloc[0])
                    total_flare_count = under_c5_flare_count + c5_and_over_count
                    timestamp_results_ret.append({"FlareTimestamp": timestamp,
                                                       "UnderC5Percentage": under_c5_flare_count / total_flare_count if total_flare_count else 0,
                                                       "C5AndOverPercentage": c5_and_over_count / total_flare_count if total_flare_count else 0,
                                                       "MeetsMinimumFlareCount": True if total_flare_count >= minimum_flare_count else False,
                                                       "GT": class_gt})
                flare_results[flare_id] = (timestamp_results_ret)
            threshold_results[('{:.2E}'.format(float(xrsb_bin_value)), '{:.2E}'.format(float(xrsb_difference_bin_value)))] = flare_results
            results.append(threshold_results)

    plt.clf()
    for result in results:
        for thresholds, flare_list in result.items():
            for flare_id, flare_results in flare_list.items():
                xs = [int(x["FlareTimestamp"]) for x in flare_results]
                strong_flare_percentage = [x["C5AndOverPercentage"] for x in flare_results]
                weak_flare_percentage = [x["UnderC5Percentage"] for x in flare_results]
                if flare_c5_or_higher(flare_results[0]["GT"]):
                    plt.plot(xs, strong_flare_percentage, linestyle="solid", color="green")
                    plt.plot(xs, weak_flare_percentage, linestyle="solid", color="red")
                else:
                    plt.plot(xs, strong_flare_percentage, linestyle="dotted", color="green")
                    plt.plot(xs, weak_flare_percentage, linestyle="dotted", color="red")

            plt.show()

# unweighted_confusion_matrix()
find_c5_threshold()

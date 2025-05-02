import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

"""Graphs binary accuracy (less/greater than C5) for all models. The graph is kinda messy, just used for high-level debugging,"""

results_dir = "../MSI Results"
run_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"

# get datasets
split_datasets_filepath = os.path.join(results_dir, run_nickname, "Datasets", "split_datasets0_minutes_since_start.pkl")
with open(split_datasets_filepath, 'rb') as f:
    split_datasets = pickle.load(f)

temporal_test_gt_and_results = pd.concat([split_datasets["temporal_test"]["additional_data"].FlareID,
                                          split_datasets["temporal_test"]["additional_data"].FlareClass,
                                          split_datasets["temporal_test"]["additional_data"]["IsFlareClass>=C5"]], axis=1)
columns = ["FlareID", "FlareClass", "IsFlareClass>=C5"]

for minutes_since_start in range(-5, 16):
    trained_tree_filepath = os.path.join(results_dir, run_nickname, "Trees", f"trained_{minutes_since_start}_minutes_since_start")
    with open(trained_tree_filepath, 'rb') as f:
        trained_tree = pickle.load(f)
    predictions = trained_tree.predict(split_datasets["temporal_test"]["x"])
    temporal_test_gt_and_results = pd.concat([temporal_test_gt_and_results, split_datasets["temporal_test"]["y"]], axis=1)
    temporal_test_gt_and_results = pd.concat([temporal_test_gt_and_results, pd.DataFrame(predictions)], axis=1)
    columns.append(f"GT{minutes_since_start}MinutesSinceStart")
    columns.append(f"Prediction{minutes_since_start}MinutesSinceStart")
temporal_test_gt_and_results.columns = columns
temporal_test_gt_and_results.sort_values("FlareID", inplace=True)

flare_accuracy_count = {}
no_flares_mask = {}
for year in range(2017, 2025):
    for month in range(1, 13):
        flare_accuracy_count[f"{year}-{month}"] = {}
        no_flares_mask[f"{year}-{month}"] = {}
        for minutes_since_start in range(-5, 16):
            flare_accuracy_count[f"{year}-{month}"][minutes_since_start] = 0.0
            no_flares_mask[f"{year}-{month}"][minutes_since_start] = True
        if month <= 8:
            this_month = str(month).zfill(2)
            next_month = str(month + 1).zfill(2)
        elif month == 9:
            this_month = str(month).zfill(2)
            next_month = str(month + 1)
        else:
            this_month = str(month)
            next_month = str(month + 1)
        subset = temporal_test_gt_and_results[(int(f"{year}{this_month}{'000000'}") <= temporal_test_gt_and_results.FlareID) & (temporal_test_gt_and_results.FlareID < int(f"{year}{next_month}{'000000'}"))]
        for minutes_since_start in range(-5, 16):
            gt_and_preds_at_time = subset.loc[:, ["FlareID", f"GT{minutes_since_start}MinutesSinceStart",
                                                  f"Prediction{minutes_since_start}MinutesSinceStart"]]
            for idx, row in gt_and_preds_at_time.iterrows():
                gt = bool(row[f"GT{minutes_since_start}MinutesSinceStart"])
                pred = bool(row[f"Prediction{minutes_since_start}MinutesSinceStart"])
                if gt == pred:
                    flare_accuracy_count[f"{year}-{month}"][minutes_since_start] += (1 / gt_and_preds_at_time.shape[0])
                    no_flares_mask[f"{year}-{month}"][minutes_since_start] = False

xs = list(flare_accuracy_count.keys())
xlabels = []
# graph!
for minutes_since_start in range(-5, 16):
    ys = []
    for year in range(2017, 2025):
        for month in range(1, 13):
            ys.append(flare_accuracy_count[f"{year}-{month}"][minutes_since_start])
            if minutes_since_start == -5:
                xlabels.append(f"{year}-{month}")
    plt.plot(xs, ys, label=minutes_since_start)

idx = 0
for year in range(2017, 2025):
    for month in range(1, 13):
        if no_flares_mask[f"{year}-{month}"][minutes_since_start]:
            plt.scatter(idx, 0, color='red', marker='o', s=75)
        idx += 1

plt.title("Adjusted Accuracy Over Time", fontsize=24)
plt.ylabel("Adjusted Accuracy", fontsize=22)
plt.xlabel("Date (YYYY-MM)", fontsize=22)
plt.xticks(ticks=range(len(xlabels))[::6], labels=xlabels[::6], fontsize=18, rotation=45)
plt.yticks(fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

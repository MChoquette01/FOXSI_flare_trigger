import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Compare several models to each other and FOXSI-4 parameter search results"""

results_base_dir = r"../MSI Results"
# datasets = ["2025_03_06_multiclass_naive_adjusted_precision_dt", "2025_03_06_multiclass_adjusted_precision_dt", "2025_03_08_multiclass_naive_adjusted_precision_rf", "2025_03_07_multiclass_adjusted_precision_rf", "2025_03_21_multiclass_naive_adjusted_precision_gbdt"]
datasets = ["2025_03_06_multiclass_naive_adjusted_precision_dt", "2025_03_08_multiclass_naive_adjusted_precision_rf", "2025_03_21_multiclass_naive_adjusted_precision_gbdt"]

all_precisions = []
all_recalls = []
column_names = []
for dataset in datasets:
    dataset_scores_path = os.path.join(results_base_dir, dataset, "Cancellation Analysis", "Cancellation Confusion Matrices", "Cancellation Effects.csv")
    scores = pd.read_csv(dataset_scores_path)
    all_precisions.append([dataset] + [x for x in scores.AdjustedPrecisionWithCancellation.T.astype(np.float64)])
    all_recalls.append([dataset] + [x for x in scores.AdjustedRecallWithCancellation.T.astype(np.float64)])
all_precisions = pd.DataFrame(np.array(all_precisions), columns=["Dataset"] + [x for x in range(-5, 16)])
all_recalls = pd.DataFrame(np.array(all_recalls), columns=["Dataset"] + [x for x in range(-5, 16)])

with pd.ExcelWriter("../test.xlsx") as writer:
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    all_precisions.to_excel(writer, sheet_name="Precision", index=False)
    all_recalls.to_excel(writer, sheet_name="Recall", index=False)

# plot
param_search_precision = 0.94
param_search_recall = 0.53


xs = [x for x in range(-5, 16)]
plt.figure(figsize=(16, 9))
for _, row in all_precisions.iterrows():
    if "gbdt" in row.Dataset:
        label = "Gradient Boosted Tree"
    elif "dt" in row.Dataset:
        label = "Decision Tree"
    elif "rf" in row.Dataset:
        label = "Random Forest"
    if "naive" in row.Dataset:
        plt.plot(xs, row.iloc[1:].astype(float), label=label)
        for idx, precision in enumerate(row.iloc[1:].astype(float)):
            if precision > param_search_precision:
                plt.scatter(idx - 5, precision, color='red', marker='o', s=75)
plt.axhline(param_search_precision, linestyle="--", color='black', label="Parameter Search Precision")

plt.title("Adjusted Precision with Cancellation", fontsize=24)
plt.xlabel("Minutes Since Flare Start", fontsize=22)
plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
plt.ylabel("Adjusted Precision", fontsize=22)
plt.yticks(ticks=np.arange(0, 1.1, 0.25), labels=np.arange(0, 1.1, 0.25), fontsize=20)
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 9))
for idx, row in all_recalls.iterrows():
    if "gbdt" in row.Dataset:
        label = "Gradient Boosted Tree"
    elif "dt" in row.Dataset:
        label= "Decision Tree"
    elif "rf" in row.Dataset:
        label = "Random Forest"
    if "naive" in row.Dataset:
        plt.plot(xs, row.iloc[1:].astype(float), label=label)
        for idx, recall in enumerate(row.iloc[1:].astype(float)):
            if recall > param_search_recall:
                plt.scatter(idx - 5, recall, color='red', marker='o', s=75)
plt.axhline(param_search_recall, linestyle="--", color='black', label="Parameter Search Recall")
plt.title("Adjusted Recall with Cancellation", fontsize=24)
plt.xlabel("Minutes Since Flare Start", fontsize=22)
plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
plt.ylabel("Adjusted Recall", fontsize=22)
plt.yticks(ticks=np.arange(0, 1.1, 0.25), labels=np.arange(0, 1.1, 0.25), fontsize=20)
plt.legend(fontsize='xx-large')
plt.tight_layout()
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import seaborn as sns

"""Main results plotting script. Plots improvements made during pruning, cancellation policy application and compares
models with non-XRS temp/EM differences to those without"""

results_dir = "../MSI Results"
decision_tree_nickname = "2025_03_06_multiclass_adjusted_precision_dt"
random_forest_nickname = "2025_03_07_multiclass_adjusted_precision_rf_msismall"
gbdt_nickname = "2025_03_06_multiclass_adjusted_precision_gbdt_debug"
naive_decision_tree_nickname = "2025_03_06_multiclass_naive_adjusted_precision_dt"
naive_random_forest_nickname = "2025_03_08_multiclass_naive_adjusted_precision_rf"
naive_gbdt_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"

all_scores_template = {"training": {"adjusted_precision": None,
                           "adjusted_recall": None,
                           "adjusted_f1": None},
              "pruning": {"adjusted_precision": None,
                           "adjusted_recall": None,
                           "adjusted_f1": None},
              "cancellation": {"adjusted_precision": None,
                               "adjusted_recall": None,
                               "adjusted_f1": None}
              }

all_scores = {"tree": deepcopy(all_scores_template),
              "random_forest": deepcopy(all_scores_template),
              "gbdt": deepcopy(all_scores_template),
              "naive_tree": deepcopy(all_scores_template),
              "naive_random_forest": deepcopy(all_scores_template),
              "naive_gbdt": deepcopy(all_scores_template)}

models = list(all_scores.keys())
formal_model_names = ["Decision Tree (XRS Differences Only)", "Random Forest (XRS Differences Only)", "Gradient Boosted Tree (XRS Differences Only)",
                      "Decision Tree", "Random Forest", "Gradient Boosted Tree"]
nicknames = [decision_tree_nickname, random_forest_nickname, gbdt_nickname,
             naive_decision_tree_nickname, naive_random_forest_nickname, naive_gbdt_nickname]
metrics = ["adjusted_precision", "adjusted_recall", "adjusted_f1"]
colors = ["lightblue", "peachpuff", "lawngreen", "dodgerblue", "darkorange", "green"]


for model_type, nickname in zip(models, nicknames):
    pruned_scores_filepath = os.path.join(results_dir, nickname, "Pruning", "Pruning Confusion Matrices", "Pruning Effects.csv")
    cancelled_scores_filepath = os.path.join(results_dir, nickname, "Cancellation Analysis", "Cancellation Confusion Matrices", "Cancellation Effects.csv")
    pruned_scores = pd.read_csv(pruned_scores_filepath)
    cancelled_scores = pd.read_csv(cancelled_scores_filepath)

    all_scores[model_type]["training"]["adjusted_precision"] = pruned_scores.AdjustedPrecisionWithoutPruning.tolist()
    all_scores[model_type]["training"]["adjusted_recall"] = pruned_scores.AdjustedRecallWithoutPruning
    all_scores[model_type]["training"]["adjusted_f1"] = pruned_scores.AdjustedF1WithoutPruning
    all_scores[model_type]["pruning"]["adjusted_precision"] = pruned_scores.AdjustedPrecisionWithPruning
    all_scores[model_type]["pruning"]["adjusted_recall"] = pruned_scores.AdjustedRecallWithPruning
    all_scores[model_type]["pruning"]["adjusted_f1"] = pruned_scores.AdjustedF1WithPruning
    all_scores[model_type]["cancellation"]["adjusted_precision"] = cancelled_scores.AdjustedPrecisionWithCancellation
    all_scores[model_type]["cancellation"]["adjusted_recall"] = cancelled_scores.AdjustedRecallWithCancellation
    all_scores[model_type]["cancellation"]["adjusted_f1"] = cancelled_scores.AdjustedF1WithCancellation


xs = [x for x in range(-5, 16)]
yticks = [x / 10 for x in range(11)]

for model_type, formal_model_name, color in zip(models, formal_model_names, colors):
    plt.figure(figsize=(16, 9))
    plt.plot(xs, all_scores[model_type]["training"]["adjusted_precision"], color='greenyellow', label="Training")
    plt.plot(xs, all_scores[model_type]["pruning"]["adjusted_precision"], color='springgreen', label="Pruning")
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_precision"], color='darkgreen', label="Cancellation")
    plt.title(f"{formal_model_name} Post-Processing Adjusted Precision Improvements", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=20)
    plt.ylabel("Adjusted Precision", fontsize=20)
    plt.xticks(ticks=xs, labels=xs, fontsize=18)
    plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(os.path.join("Saved Plots", f"{formal_model_name} Adjusted Precision Stages.png"))
    # plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(xs, all_scores[model_type]["training"]["adjusted_recall"], color='greenyellow', label="Training")
    plt.plot(xs, all_scores[model_type]["pruning"]["adjusted_recall"], color='springgreen', label="Pruning")
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_recall"], color='darkgreen', label="Cancellation")
    plt.title(f"{formal_model_name} Post-Processing Adjusted Recall Improvements", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=20)
    plt.ylabel("Adjusted Recall", fontsize=20)
    plt.xticks(ticks=xs, labels=xs, fontsize=18)
    plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(os.path.join("Saved Plots", f"{formal_model_name} Adjusted Recall Stages.png"))
    # plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(xs, all_scores[model_type]["training"]["adjusted_f1"], color='greenyellow', label="Training")
    plt.plot(xs, all_scores[model_type]["pruning"]["adjusted_f1"], color='springgreen', label="Pruning")
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_f1"], color='darkgreen', label="Cancellation")
    plt.title(f"{formal_model_name} Post-Processing Adjusted F1 Improvements", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=20)
    plt.ylabel("Adjusted F1", fontsize=20)
    plt.xticks(ticks=xs, labels=xs, fontsize=18)
    plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(os.path.join("Saved Plots", f"{formal_model_name} Adjusted F1 Stages.png"))
    # plt.show()

# all final models comparison
plt.figure(figsize=(16, 9))
for model_type, formal_model_name, color in zip(models, formal_model_names, colors):
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_precision"], color=color, label=formal_model_name)
plt.title("Final Models Adjusted Precision", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("Adjusted Precision", fontsize=20)
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize='xx-large')
plt.savefig(os.path.join("Saved Plots", "Adjusted Precision Comparison.png"))
# plt.show()

plt.figure(figsize=(16, 9))
for model_type, formal_model_name, color in zip(models, formal_model_names, colors):
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_recall"], color=color, label=formal_model_name)
plt.title("Final Models Adjusted Recall", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("Adjusted Recall", fontsize=20)
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize='xx-large')
plt.savefig(os.path.join("Saved Plots", "Adjusted Recall Comparison.png"))
# plt.show()

plt.figure(figsize=(16, 9))
for model_type, formal_model_name, color in zip(models, formal_model_names, colors):
    plt.plot(xs, all_scores[model_type]["cancellation"]["adjusted_f1"], color=color, label=formal_model_name)
plt.title("Final Models Adjusted F1", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("Adjusted F1", fontsize=20)
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize='xx-large')
plt.savefig(os.path.join("Saved Plots", "Adjusted Precision F1.png"))
# plt.show()

# XRS v. non_XRS differences
# https://stackoverflow.com/questions/67908107/how-to-create-a-legend-separated-by-color-and-linestyle

# load the data from the OP in to the dict
dt_data = {'x': np.concatenate((xs, xs, xs, xs, xs, xs)),
        'vals': np.concatenate((all_scores["tree"]["cancellation"]["adjusted_precision"],
                                all_scores["tree"]["cancellation"]["adjusted_recall"],
                                all_scores["tree"]["cancellation"]["adjusted_f1"],
                                all_scores["naive_tree"]["cancellation"]["adjusted_precision"],
                                all_scores["naive_tree"]["cancellation"]["adjusted_recall"],
                                all_scores["naive_tree"]["cancellation"]["adjusted_f1"])),
        'Model': ['Decision Tree (XRS Differences Only)']*len(all_scores["tree"]["cancellation"]["adjusted_precision"]) +
               ['Decision Tree (XRS Differences Only)']*len(all_scores["tree"]["cancellation"]["adjusted_recall"]) +
               ['Decision Tree (XRS Differences Only)']*len(all_scores["tree"]["cancellation"]["adjusted_f1"]) +
               ['Decision Tree']*len(all_scores["naive_tree"]["cancellation"]["adjusted_precision"]) +
               ['Decision Tree']*len(all_scores["naive_tree"]["cancellation"]["adjusted_recall"]) +
               ['Decision Tree']*len(all_scores["naive_tree"]["cancellation"]["adjusted_f1"]),
        'Metric': ['Adjusted Precision']*len(all_scores["tree"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["tree"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["tree"]["cancellation"]["adjusted_f1"]) +
                ['Adjusted Precision']*len(all_scores["naive_tree"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["naive_tree"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["naive_tree"]["cancellation"]["adjusted_f1"])}

rf_data = {'x': np.concatenate((xs, xs, xs, xs, xs, xs)),
        'vals': np.concatenate((all_scores["random_forest"]["cancellation"]["adjusted_precision"],
                                all_scores["random_forest"]["cancellation"]["adjusted_recall"],
                                all_scores["random_forest"]["cancellation"]["adjusted_f1"],
                                all_scores["naive_random_forest"]["cancellation"]["adjusted_precision"],
                                all_scores["naive_random_forest"]["cancellation"]["adjusted_recall"],
                                all_scores["naive_random_forest"]["cancellation"]["adjusted_f1"])),
        'Model': ['Random Forest (XRS Differences Only)']*len(all_scores["random_forest"]["cancellation"]["adjusted_precision"]) +
               ['Random Forest (XRS Differences Only)']*len(all_scores["random_forest"]["cancellation"]["adjusted_recall"]) +
               ['Random Forest (XRS Differences Only)']*len(all_scores["random_forest"]["cancellation"]["adjusted_f1"]) +
               ['Random Forest']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_precision"]) +
               ['Random Forest']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_recall"]) +
               ['Random Forest']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_f1"]),
        'Metric': ['Adjusted Precision']*len(all_scores["random_forest"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["random_forest"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["random_forest"]["cancellation"]["adjusted_f1"]) +
                ['Adjusted Precision']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["naive_random_forest"]["cancellation"]["adjusted_f1"])}

gbdt_data = {'x': np.concatenate((xs, xs, xs, xs, xs, xs)),
        'vals': np.concatenate((all_scores["gbdt"]["cancellation"]["adjusted_precision"],
                                all_scores["gbdt"]["cancellation"]["adjusted_recall"],
                                all_scores["gbdt"]["cancellation"]["adjusted_f1"],
                                all_scores["naive_gbdt"]["cancellation"]["adjusted_precision"],
                                all_scores["naive_gbdt"]["cancellation"]["adjusted_recall"],
                                all_scores["naive_gbdt"]["cancellation"]["adjusted_f1"])),
        'Model': ['Gradient Boosted Tree (XRS Differences Only)']*len(all_scores["gbdt"]["cancellation"]["adjusted_precision"]) +
               ['Gradient Boosted Tree (XRS Differences Only)']*len(all_scores["gbdt"]["cancellation"]["adjusted_recall"]) +
               ['Gradient Boosted Tree (XRS Differences Only)']*len(all_scores["gbdt"]["cancellation"]["adjusted_f1"]) +
               ['Gradient Boosted Tree']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_precision"]) +
               ['Gradient Boosted Tree']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_recall"]) +
               ['Gradient Boosted Tree']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_f1"]),
        'Metric': ['Adjusted Precision']*len(all_scores["gbdt"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["gbdt"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["gbdt"]["cancellation"]["adjusted_f1"]) +
                ['Adjusted Precision']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_precision"]) +
                ['Adjusted Recall']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_recall"]) +
                ['Adjusted F1']*len(all_scores["naive_gbdt"]["cancellation"]["adjusted_f1"])}

# plot the data
plt.figure(figsize=(16, 9))
p = sns.lineplot(data=dt_data, x='x', y='vals', hue='Model', style='Metric', palette=['teal', 'darkorange'])
plt.title("Decision Tree", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("")
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
# move the legend
p.legend(bbox_to_anchor=(1.05, 1), fontsize='xx-large', fancybox=True, shadow=True)
plt.tight_layout()
# plt.savefig(os.path.join("Saved Plots", f"Decision Tree Naive Comparison.png"))
plt.show()

plt.figure(figsize=(16, 9))
p = sns.lineplot(data=rf_data, x='x', y='vals', hue='Model', style='Metric', palette=['teal', 'darkorange'])
plt.title("Random Forest", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("")
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
# move the legend
p.legend(bbox_to_anchor=(1.05, 1), fontsize='xx-large', fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join("Saved Plots", f"Random Forest Naive Comparison.png"))
# plt.show()

plt.figure(figsize=(16, 9))
p = sns.lineplot(data=gbdt_data, x='x', y='vals', hue='Model', style='Metric', palette=['teal', 'darkorange'])
plt.title("Gradient Boosted Tree", fontsize=22)
plt.xlabel("Minutes Since Flare Start", fontsize=20)
plt.ylabel("")
plt.xticks(ticks=xs, labels=xs, fontsize=18)
plt.yticks(ticks=yticks, labels=yticks, fontsize=18)
# move the legend
p.legend(bbox_to_anchor=(1.05, 1), fontsize='xx-large', fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join("Saved Plots", f"Gradient Boosted Tree Naive Comparison.png"))
# plt.show()




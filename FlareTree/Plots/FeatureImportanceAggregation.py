import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.insert(0, "../")
import tree_common as tc

"""Aggregate feature importance from all models into 2 bins; pre-flare (1-5 minutes BEFORE flare start) and
early flare (0-2 minutes AFTER start). Truncating early avoids a potential mix of impulsive/gradual phases."""

results_dir = "../MSI Results"
decision_tree_nickname = "2025_03_06_multiclass_adjusted_precision_dt"
random_forest_nickname = "2025_03_07_multiclass_adjusted_precision_rf_msismall"
gbdt_nickname = "2025_03_06_multiclass_adjusted_precision_gbdt_debug"
naive_decision_tree_nickname = "2025_03_06_multiclass_naive_adjusted_precision_dt"
naive_random_forest_nickname = "2025_03_08_multiclass_naive_adjusted_precision_rf"
naive_gbdt_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"

all_imporatances = {"tree": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)},
              "random_forest": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)},
              "gbdt": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)},
              "naive_tree": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)},
              "naive_random_forest": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)},
              "naive_gbdt": {"pre-flare": defaultdict(list), "early_flare": defaultdict(list)}}

models = list(all_imporatances.keys())
formal_model_names = ["Decision Tree (XRS Differences Only)", "Random Forest (XRS Differences Only)", "Gradient Boosted Tree (XRS Differences Only)",
                      "Decision Tree", "Random Forest", "Gradient Boosted Tree"]
nicknames = [decision_tree_nickname, random_forest_nickname, gbdt_nickname,
             naive_decision_tree_nickname, naive_random_forest_nickname, naive_gbdt_nickname]

# pre-flare stage (5 to 1 minutes before flare start)
for minutes_since_start in range(-5, 0):
    for nickname, model_type in zip(nicknames, models):
        feature_importance_filepath = os.path.join(results_dir, nickname, "Feature Importance", f"feature_importances_{minutes_since_start + 15}_minutes_since_flare_start.pkl")
        with open(feature_importance_filepath, 'rb') as f:
            feature_importances = pickle.load(f)
        for feature_importance in feature_importances:
            feature, importance = feature_importance[0], feature_importance[1]
            all_imporatances[model_type]["pre-flare"][tc.FORMAL_VARIABLE_NAMES[feature]].append(importance)

# early flare stage (0 to 2 minutes after flare start)
for minutes_since_start in range(0, 3):
    for nickname, model_type in zip(nicknames, models):
        feature_importance_filepath = os.path.join(results_dir, nickname, "Feature Importance", f"feature_importances_{minutes_since_start + 15}_minutes_since_flare_start.pkl")
        with open(feature_importance_filepath, 'rb') as f:
            feature_importances = pickle.load(f)
        for feature_importance in feature_importances:
            feature, importance = feature_importance[0], feature_importance[1]
            all_imporatances[model_type]["early_flare"][tc.FORMAL_VARIABLE_NAMES[feature]].append(importance)

# change values from lists to average of lists = average feature importance over range
for model_type in models:
    for feature, importances in all_imporatances[model_type]["pre-flare"].items():
        all_imporatances[model_type]["pre-flare"][feature] = sum(importances) / len(importances)
    for feature, importances in all_imporatances[model_type]["early_flare"].items():
        all_imporatances[model_type]["early_flare"][feature] = sum(importances) / len(importances)

for model_type, model_name in zip(models, formal_model_names):
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    all_imporatances[model_type]["pre-flare"] = sorted(all_imporatances[model_type]["pre-flare"].items(), key=lambda x: x[1])
    ax[0].barh([x for x in range(len(all_imporatances[model_type]["pre-flare"]))], [x[1] for x in all_imporatances[model_type]["pre-flare"]], color='#1f77b4')
    ax[0].set_yticks(ticks=[x for x in range(len(all_imporatances[model_type]["pre-flare"]))], labels=[x[0] for x in all_imporatances[model_type]["pre-flare"]], fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[0].set_ylabel("Feature", fontsize=20)
    ax[0].set_title("Pre-Flare", fontsize=24)

    textstr = '\n'.join((r'   1 - 5 minutes', r'before flare start'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[0].text(0.25, 0.3, textstr, transform=ax[0].transAxes, fontsize=22,
            verticalalignment='top', bbox=props)

    all_imporatances[model_type]["early_flare"] = sorted(all_imporatances[model_type]["early_flare"].items(), key=lambda x: x[1])
    for idx, feature in enumerate(all_imporatances[model_type]["pre-flare"]):
        list_idx = [x for x in all_imporatances[model_type]["early_flare"] if x[0] == feature[0]][0]
        ax[1].barh(idx, list_idx[1], color='#1f77b4')
    ax[1].set_yticks(ticks=[x for x in range(len(all_imporatances[model_type]["early_flare"]))], labels=[x[0] for x in all_imporatances[model_type]["early_flare"]], fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    fig.text(0.6, 0.04, "Normalized Total Reduction of Split Criteria (Averaged)", ha='center', fontsize=20)
    ax[1].set_title("Early Flare", fontsize=24)

    plt.suptitle(f"{model_name} Feature Importance", fontsize=26)
    plt.subplots_adjust(left=0.35)

    textstr = '\n'.join((r'  0 - 2 minutes', r'after flare start'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.25, 0.3, textstr, transform=ax[1].transAxes, fontsize=22,
            verticalalignment='top', bbox=props)

    plt.savefig(os.path.join("Saved Plots", f"{model_name} Average Feature Importance.png"))
    # plt.show()
    plt.close()
    plt.clf()
    plt.cla()




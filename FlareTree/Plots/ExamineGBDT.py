import os
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

"""Plot an individual tree from a Gradient Boosted tree (and probably random forests too!)"""

results_dir = "../MSI Results"
run_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"
sub_tree_index = 32

for minutes_since_start in range(-5, 16):
    # get data and model
    split_datasets_filepath = os.path.join(results_dir, run_nickname, "Datasets", f"split_datasets{minutes_since_start}_minutes_since_start.pkl")
    trained_tree_filepath = os.path.join(results_dir, run_nickname, "Trees", f"trained_{minutes_since_start}_minutes_since_start")
    with open(trained_tree_filepath, 'rb') as f:
        trained_tree = pickle.load(f)
    with open(split_datasets_filepath, 'rb') as f:
        split_datasets = pickle.load(f)
    plot_tree(trained_tree.estimators_[sub_tree_index][0], feature_names=split_datasets["train"]["x"].columns,
              class_names=[f"< C5", f"<= C5 x < M0", f"<= M0 x < X0", f">= X0"], filled=True, proportion=True,
              rounded=True, precision=9, fontsize=10)
    plt.show()

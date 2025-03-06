from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os
import pickle
import  tree_common as tc

results_filepath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results\results.pkl"

out_dir = os.path.join(os.path.split(results_filepath)[0], "Tree Graphs", "Bigger")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(results_filepath, "rb") as f:
    results = pickle.load(f)

for timestamp in results.minutes_since_start.tolist():
    t = tc.create_tree_from_df(results, int(timestamp))
    split_datasets = tc.get_train_and_test_data_from_pkl(int(timestamp))
    split_datasets["train"]["x"], split_datasets["test"]["x"] = tc.impute_variable_data(split_datasets["train"]["x"], split_datasets["test"]["x"])

    split_datasets["train"]["x"].columns = ["CurrentXSRA", "XRSA1MinuteDifference", "XRSA3MinuteDifference",
                                            "XRSA5MinuteDifference", "CurrentXRSB", "XRSB1MinuteDifference", "XRSB3MinuteDifference",
                                            "XRSB5MinuteDifference", "Temperature", "Temperature1MinuteDifference",
                                            "Temperature3MinuteDifference", "Temperature5MinuteDifference", "EmissionMeasure",
                                            "EmissionMeasure1MinuteDifference", "EmissionMeasure3MinuteDifference",
                                            "EmissionMeasure5MinuteDifference"]

    t.fit(split_datasets["train"]["x"], split_datasets["train"]["y"])

    # plot best tree structure
    plt.figure(figsize=(200, 113))  # big, so rectangles don't overlap
    plot_tree(t, feature_names=split_datasets["train"]["x"].columns, class_names=["<C5", ">=C5"], filled=True, proportion=True, rounded=True, precision=9, fontsize=10)
    graph_out_path = os.path.join(os.path.split(results_filepath)[0], "Tree Graphs", "Bigger", f"{timestamp}_minutes_since_start_tree.png")
    plt.savefig(graph_out_path)


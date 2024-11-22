import numpy as np
import pickle
import tree_common as tc
import os
from tqdm import tqdm

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

"""Fun functions to see a path a record takes through a tree.
Pretty shamelessly adapted from here:
https://stackoverflow.com/questions/57326537/scikit-learn-decision-tree-extract-nodes-for-feature"""


def print_tree_structure(t, feature_names):
    """Pretty-print tree structure"""

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node

    n_nodes = t.tree_.node_count
    children_left = t.tree_.children_left
    children_right = t.tree_.children_right
    feature = t.tree_.feature
    threshold = t.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if %s <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature_names[feature[i]],
                     threshold[i],
                     children_right[i],
                     ))
    print("\n")


def get_path(t, test_x, sample_id):

    feature = t.tree_.feature
    threshold = t.tree_.threshold
    values = t.tree_.value

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.
    node_indicator = t.decision_path(test_x)

    # Similarly, we can also have the leaves ids reached by each sample.
    leave_id = t.apply(test_x)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    is_strong_flare = bool(test_y.iloc[sample_id].iloc[0])  # gt
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:

        if leave_id[sample_id] == node_id:  # <-- changed != to ==
            #continue # <-- comment out
            print("Leaf node {} reached".format(leave_id[sample_id])) # <--
            node_values = values[node_id, :, :][0]
            is_predicted_strong_flare = True if node_values[1] > node_values[0] else False
            if is_predicted_strong_flare:
                confidence = node_values[1] / (node_values[1] + node_values[0]) * 100
            else:
                confidence = node_values[0] / (node_values[1] + node_values[0]) * 100
            print(f"\nIs >=C5 flare (GT): {is_strong_flare}\nPredicted as a >=C5 flare: {is_predicted_strong_flare}\n"
                  f"Confidence: {confidence}")

        else: # < -- added else to iterate through decision nodes
            if (test_x.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("Decision node %s : (X[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     train_x.columns[feature[node_id]],
                     test_x.iloc[sample_id, feature[node_id]], # <-- changed i to sample_id
                     threshold_sign,
                     threshold[node_id]))


def get_path_stats(t, x, y):

    feature = t.tree_.feature
    threshold = t.tree_.threshold
    values = t.tree_.value

    node_indicator = t.decision_path(x)

    leave_id = t.apply(x)

    confidences_by_outcome = {"true_positive": [], "true_negative": [], "false_positive": [], "false_negative": []}
    for sample_id in range(x.shape[0]):
        is_strong_flare = bool(y.iloc[sample_id].iloc[0])  # gt
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:

            if leave_id[sample_id] == node_id:  # <-- changed != to ==
                node_values = values[node_id, :, :][0]
                is_predicted_strong_flare = True if node_values[1] > node_values[0] else False
                if is_predicted_strong_flare:
                    confidence = node_values[1] / (node_values[1] + node_values[0]) * 100
                else:
                    confidence = node_values[0] / (node_values[1] + node_values[0]) * 100

                if is_strong_flare:
                    if is_predicted_strong_flare:
                        confidences_by_outcome["true_positive"].append(confidence)
                    else:
                        confidences_by_outcome["false_negative"].append(confidence)
                else:
                    if is_predicted_strong_flare:
                        confidences_by_outcome["false_positive"].append(confidence)
                    else:
                        confidences_by_outcome["true_negative"].append(confidence)

            else: # < -- added else to iterate through decision nodes
                continue

    return confidences_by_outcome


def get_confidence_graph(t, train_x, test_x, train_y, test_y, minutes_since_start):

    train_confidence_values = get_path_stats(t, train_x, train_y)
    test_confidence_values = get_path_stats(t, test_x, test_y)

    conf_array = np.array([[train_confidence_values["true_positive"]], train_confidence_values["true_negative"], train_confidence_values["false_negative"], train_confidence_values["true_negative"]])
    plt.figure(figsize=(16,9))
    plt.hist(conf_array, 100, density=False, histtype='bar', stacked=True, color=["green", "springgreen", "lightsalmon", "red"], label=["TP", "TN", "FN", "FP"])
    plt.title('Prediction Confidences')
    plt.xlabel("Confidence of Prediction")
    plt.ylabel("Count")
    plt.title(f"{minutes_since_start - 15} minutes before flare start")
    plt.legend()
    plt.savefig(os.path.join("Analysis", f"ConfidenceGraph_{minutes_since_start}_minutes_to_start.png"))
    # plt.show()


def get_science_goals_accuracy(test_x_flare_ids, test_x, test_y, minutes_since_start):

    results = {"TP": {"Observed": 0, "Not observed": 0},
               "FP": {"Observed": 0, "Not observed": 0},
               "FN": {"Observed": 0, "Not observed": 0},
               "TN": {"Observed": 0, "Not observed": 0}}
    LAUNCH_DELAY_MINUTES = 7
    OBSERVATION_TIME_MINUTES = 6
    flare_ids = [int(x) for x in test_x_flare_ids.to_list()]
    test_y = test_y.IsC5OrHigher.to_list()
    test_predictions = t.predict(test_x.values)
    # get all records from this timestamp
    flare_minutes_to_peak = {}
    client, flares_table = tc.connect_to_flares_db()
    cursor = flares_table.find({"FlareID": {"$regex": f"_{minutes_since_start}$"}})
    for record in cursor:
        flare_minutes_to_peak[int(record["FlareID"].split("_")[0])] = record["MinutesToPeak"]
    client.close()
    for flare_id, gt, prediction in tqdm(zip(flare_ids, test_y, test_predictions)):
        minutes_to_peak = flare_minutes_to_peak[flare_id]
        if gt:
            if prediction:
                if LAUNCH_DELAY_MINUTES <= minutes_to_peak <= LAUNCH_DELAY_MINUTES + OBSERVATION_TIME_MINUTES:
                    results["TP"]["Observed"] += 1
                else:
                    results["TP"]["Not observed"] += 1
            else:
                if LAUNCH_DELAY_MINUTES <= minutes_to_peak <= LAUNCH_DELAY_MINUTES + OBSERVATION_TIME_MINUTES:
                    results["FN"]["Observed"] += 1
                else:
                    results["FN"]["Not observed"] += 1
        else:
            if prediction:
                if LAUNCH_DELAY_MINUTES <= minutes_to_peak <= LAUNCH_DELAY_MINUTES + OBSERVATION_TIME_MINUTES:
                    results["FP"]["Observed"] += 1
                else:
                    results["FP"]["Not observed"] += 1
            else:
                if LAUNCH_DELAY_MINUTES <= minutes_to_peak <= LAUNCH_DELAY_MINUTES + OBSERVATION_TIME_MINUTES:
                    results["TN"]["Observed"] += 1
                else:
                    results["TN"]["Not observed"] += 1
    print(results)


results_folderpath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\MSI Results"
run_nickname = ""

out_dir = os.path.join(results_folderpath, "Analysis")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

results = tc.get_results_pickle(results_folderpath, run_nickname)

for timestamp in results.minutes_since_start.tolist()[:1]:
    t = tc.create_tree_from_df(results, int(timestamp))
    train_x, train_x_flare_ids, train_y, test_x, test_x_flare_ids, test_y = tc.get_train_and_test_data_from_pkl(int(timestamp))
    train_x, test_x = tc.impute_variable_data(train_x, test_x)
    t.fit(train_x, train_y)
    # print_tree_structure(t, list(train_x.columns))
    # get_path(t, test_x, sample_id=234)
    # get_confidence_graph(t, train_x, test_x, train_y, test_y, int(timestamp))
    get_science_goals_accuracy(test_x_flare_ids, test_x, test_y, int(timestamp))

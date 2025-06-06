import numpy as np
import pandas as pd
import pickle
import re
import tree_common as tc
import os
from tqdm import tqdm

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

"""Fun functions to see a path a record takes through a tree.
Pretty shamelessly adapted from here:
https://stackoverflow.com/questions/57326537/scikit-learn-decision-tree-extract-nodes-for-feature

Many of these functions will only work for decision trees, not random forests or gradient boosted trees"""


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
    is_strong_flare = bool(split_datasets["train"]["y"].iloc[sample_id].iloc[0])  # gt
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
                     split_datasets["train"]["x"].columns[feature[node_id]],
                     split_datasets["test"]["x"].iloc[sample_id, feature[node_id]], # <-- changed i to sample_id
                     threshold_sign,
                     threshold[node_id]))


def get_path_stats(t, x, y):

    confidences_by_outcome = {"true_positive": [], "true_negative": [], "false_positive": [], "false_negative": []}

    if x.shape[0] == 0:  # if no flares
        return confidences_by_outcome

    feature = t.tree_.feature
    threshold = t.tree_.threshold
    values = t.tree_.value
    node_indicator = t.decision_path(x)


    leave_id = t.apply(x)

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


def get_flare_class():
    """Return the GOES class for each flare in the MongoDB"""

    client, flares_table = tc.connect_to_flares_db()

    filter = {'FlareID': re.compile(r"_0$")}
    project = {'FlareID': 1, 'FlareClass': 1}
    cursor = client['Flares']['Flares'].find(filter=filter, projection=project)
    flare_ids_and_mags = {}
    for result in cursor:
        flare_ids_and_mags[result['FlareID'].split('_')[0]] = result['FlareClass']

    client.close()

    return flare_ids_and_mags


def get_confidence_graph(t, test_x, test_x_additional_flare_data, test_y, minutes_since_start):
    """Plot histograms showing the confidences in predictions for each target class"""

    def confidence_graph_helper(plot_int, test_confidence_values, flare_class_letter):
        """Graph each subplot (target class)"""

        plt.subplot(plot_int)
        plt.xlim(50, 100)
        conf_array = np.array([test_confidence_values["true_positive"], test_confidence_values["true_negative"],
                               test_confidence_values["false_negative"], test_confidence_values["false_positive"]])
        if conf_array.size != 0:  # if some data
            plt.hist(conf_array, 50, range=(50, 100), density=False, histtype='barstacked', stacked=True,
                     color=["green", "springgreen", "lightsalmon", "red"], label=["TP", "TN", "FN", "FP"])
        plt.xlabel("Confidence of Prediction")
        plt.ylabel("Count")
        plt.title(f"{flare_class_letter} Class", fontsize=18)
        plt.legend()

    # isolate train/test data by flare magnitude (letter only)
    test_data_by_class = {}
    test_and_additional_x_data = pd.concat([test_x.reset_index(drop=True),
                                            test_x_additional_flare_data.reset_index(drop=True),
                                            test_y.reset_index(drop=True)], ignore_index=False, axis=1)
    for flare_mag_letter in ["B", "C", "M", "X"]:
        subset = test_and_additional_x_data[test_and_additional_x_data.FlareClass.str.startswith(flare_mag_letter)]
        class_test_y = subset.IsStrongFlare
        subset = subset.drop(list(test_x_additional_flare_data.columns) + ["IsStrongFlare"], axis=1)
        test_data_by_class[flare_mag_letter] = {"test_x": subset, "test_y": class_test_y}

    # get confidence value for predictions from each class
    test_confidence_values_b = get_path_stats(t, pd.DataFrame(test_data_by_class["B"]['test_x']), pd.DataFrame(test_data_by_class["B"]['test_y']))
    test_confidence_values_c = get_path_stats(t, pd.DataFrame(test_data_by_class["C"]['test_x']), pd.DataFrame(test_data_by_class["C"]['test_y']))
    test_confidence_values_m = get_path_stats(t, pd.DataFrame(test_data_by_class["M"]['test_x']), pd.DataFrame(test_data_by_class["M"]['test_y']))
    test_confidence_values_x = get_path_stats(t, pd.DataFrame(test_data_by_class["X"]['test_x']), pd.DataFrame(test_data_by_class["X"]['test_y']))

    plt.figure(figsize=(16, 9))
    confidence_graph_helper(221, test_confidence_values_b, "B")
    confidence_graph_helper(222, test_confidence_values_c, "C")
    confidence_graph_helper(223, test_confidence_values_m, "M")
    confidence_graph_helper(224, test_confidence_values_x, "X")

    plt.suptitle(f"{minutes_since_start - 15} minutes since flare start", fontsize=22)
    plt.tight_layout()

    out_directory = os.path.join(out_dir, 'Confidence Graphs')
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    plt.savefig(os.path.join(out_directory, f"ConfidenceGraph_{minutes_since_start - 15}_minutes_since_start.png"))
    # plt.show()


results_folderpath = r"C:\Users\matth\Documents\Capstone\FOXSI_flare_trigger\FlareTree\Results"
run_nickname = "2025_02_02_stratify_test"

inputs = tc.get_inputs_dict(results_folderpath, run_nickname)
peak_filtering_minutes = inputs['peak_filtering_threshold_minutes']
strong_flare_threshold = inputs['strong_flare_threshold']
use_naive_diffs = inputs['use_naive_diffs']
use_science_delay = inputs["use_science_delay"]
stratify = inputs["stratify"]

out_dir = os.path.join(results_folderpath, run_nickname)

results = tc.get_results_pickle(results_folderpath, run_nickname)

for timestamp in results.minutes_since_start.tolist():
    t = tc.get_tree(out_dir, timestamp, trained=True)
    split_datasets = tc.get_train_and_test_data_from_pkl(int(timestamp),
                                                                                                                                        strong_flare_threshold,
                                                                                                                                        use_naive_diffs=use_naive_diffs,
                                                                                                                                        peak_filtering_minutes=peak_filtering_minutes,
                                                                                                                                        stratify=stratify,
                                                                                                                                        use_science_delay=use_science_delay)
    # print_tree_structure(t, list(split_datasets["train"]["x"].columns))
    # get_path(t, split_datasets["test"]["x"], sample_id=234)
    get_confidence_graph(t, split_datasets["test"]["x"], split_datasets["test"]["additional_data"], split_datasets["test"]["y"], int(timestamp))

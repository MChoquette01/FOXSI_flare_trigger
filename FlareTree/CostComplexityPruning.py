import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import argparse
from multiprocessing import Pool
from itertools import repeat
import tree_common as tc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def graph_confusion_matrices(gt, old_preds, new_preds, results_folderpath, run_nickname, time_minutes):
    """Helper function to plot confusion matricies with and without cancellation critieria applied"""

    fig, ax = plt.subplots(1, 2, clear=True)
    fig.set_figwidth(30)
    fig.set_figheight(17)
    ax[0].set_title("Without Pruning", fontsize=24)
    ax[1].set_title("With Pruning", fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=22)
    ax[1].tick_params(axis='both', which='major', labelsize=22)
    plt.rcParams.update({'font.size': 24})

    old_display_labels = [f"C, <C5", f"C, >=C5", f"M", f"X"][:len(list(set(gt + old_preds)))]
    new_display_labels = [f"C, <C5", f"C, >=C5", f"M", f"X"][:len(list(set(gt + new_preds)))]

    ConfusionMatrixDisplay.from_predictions(gt, old_preds, display_labels=old_display_labels).plot(ax=ax[0])
    ConfusionMatrixDisplay.from_predictions(gt, new_preds, display_labels=new_display_labels).plot(ax=ax[1])

    ax[0].set_xlabel("Prediction", fontsize=24)
    ax[1].set_xlabel("Prediction", fontsize=24)
    ax[0].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)
    ax[1].set_ylabel("True Maximum Observable XRSB Flux", fontsize=24)

    if time_minutes < 0:
        fig.suptitle(f"Flare Start - {abs(time_minutes)} Minutes", fontsize=30)
    else:
        fig.suptitle(f"Flare Start + {abs(time_minutes)} Minutes", fontsize=30)
    plt.tight_layout()
    fig.savefig(os.path.join(results_folderpath, run_nickname, "Pruning", "Pruning Confusion Matrices", f"Confusion Matrix {time_minutes} Minutes Since Start.png"))
    # plt.show()
    plt.close(fig)
    plt.clf()
    plt.cla()


def make_dir_safe(folder_path):
    """MSI sometimes fails on folder creation when GridSearch.py is being run in parallel"""
    try:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    except FileExistsError:
        pass

def adj_precision(cm):
    tp = cm[1:4, 1:4].sum()
    fp = sum(cm[0, 1:4])
    return tp / (tp + fp)


def adj_recall(cm):

    tp = cm[1:4, 1:4].sum()
    fn = sum(cm[1:4, 0])
    return tp / (tp + fn)


def adj_f1(adj_p, adj_r):

    return (2 * adj_p * adj_r) / (adj_p + adj_r)


def ccp_alpha_prune(run_nickname, model_type, minutes_since_start):

    results_folderpath = r"Results"

    out_dir = os.path.join(results_folderpath, run_nickname, "Pruning")
    graphs_dir = os.path.join(results_folderpath, run_nickname, "Pruning", "Pruning Confusion Matrices")
    models_dir = os.path.join(results_folderpath, run_nickname, "Pruning", "Pruned Models")
    scores_dir = os.path.join(results_folderpath, run_nickname, "Pruning", "Scores")
    if not os.path.exists(out_dir):
        make_dir_safe((out_dir))
    if not os.path.exists(graphs_dir):
        make_dir_safe((graphs_dir))
    if not os.path.exists(models_dir):
        make_dir_safe((models_dir))
    if not os.path.exists(scores_dir):
        make_dir_safe((scores_dir))

    pruned_scores = {}
    out_filepath = os.path.join(results_folderpath, run_nickname, "Pruning", f"PruningResults_{run_nickname}.txt")

    trained_tree_filepath = os.path.join(results_folderpath, run_nickname, "Trees", f"trained_{minutes_since_start}_minutes_since_start")
    split_datasets_filepath = os.path.join(results_folderpath, run_nickname, "Datasets", f"split_datasets{minutes_since_start}_minutes_since_start.pkl")

    with open(trained_tree_filepath, "rb") as f:
        trained_tree = pickle.load(f)

    with open(split_datasets_filepath, "rb") as f:
        split_datasets = pickle.load(f)

    ccp_alphas = [1*10**-3, 2.5*10**-3, 5*10**-3, 7.5*10**-3,
                  1*10**-4, 2.5*10**-4, 5*10**-4, 7.5*10**-4,
                  1*10**-5, 2.5*10**-5, 5*10**-5, 7.5*10**-5,
                  1*10**-6, 2.5*10**-6, 5*10**-6, 7.5*10**-6,
                  1*10**-7, 2.5*10**-7, 5*10**-7, 7.5*10**-7,
                  1*10**-8, 2.5*10**-8, 5*10**-8, 7.5*10**-8,
                  1*10**-9, 2.5*10**-9, 5*10**-9, 7.5*10**-9]

    best_ccp_alpha = -1
    best_adj_precision = -1
    best_model = None
    for ccp_alpha in tqdm(ccp_alphas):
        if model_type == "Gradient Boosted Tree":
            new_model = GradientBoostingClassifier(loss=trained_tree.loss,
                                                     learning_rate=trained_tree.learning_rate,
                                                     n_estimators=trained_tree.n_estimators,
                                                     criterion=trained_tree.criterion,
                                                     max_depth=trained_tree.max_depth,
                                                     max_features=trained_tree.max_features,
                                                     min_samples_leaf=trained_tree.min_samples_leaf,
                                                     min_samples_split=trained_tree.min_samples_split,
                                                     min_weight_fraction_leaf=trained_tree.min_weight_fraction_leaf,
                                                     ccp_alpha=ccp_alpha,
                                                     random_state=tc.RANDOM_STATE)
        elif model_type == "Random Forest":
            new_model = RandomForestClassifier(n_estimators=trained_tree.n_estimators,
                                                     criterion=trained_tree.criterion,
                                                     max_depth=trained_tree.max_depth,
                                                     max_features=trained_tree.max_features,
                                                     min_samples_leaf=trained_tree.min_samples_leaf,
                                                     min_samples_split=trained_tree.min_samples_split,
                                                     min_weight_fraction_leaf=trained_tree.min_weight_fraction_leaf,
                                                     class_weight=trained_tree.class_weight,
                                                     ccp_alpha=ccp_alpha,
                                                     random_state=tc.RANDOM_STATE)
        elif model_type == "Decision Tree":
            new_model = DecisionTreeClassifier(criterion=trained_tree.criterion,
                                                     max_depth=trained_tree.max_depth,
                                                     max_features=trained_tree.max_features,
                                                     min_samples_leaf=trained_tree.min_samples_leaf,
                                                     min_samples_split=trained_tree.min_samples_split,
                                                     min_weight_fraction_leaf=trained_tree.min_weight_fraction_leaf,
                                                     class_weight=trained_tree.class_weight,
                                                     ccp_alpha=ccp_alpha,
                                                     random_state=tc.RANDOM_STATE)

        new_model.fit(split_datasets["train"]["x"], split_datasets["train"]["y"])
        test_predictions = new_model.predict(split_datasets["test"]["x"])
        test_cm = confusion_matrix(split_datasets["test"]["y"], test_predictions)
        if adj_precision(test_cm) > best_adj_precision:
            best_adj_precision = adj_precision(test_cm)
            best_ccp_alpha = ccp_alpha
            best_model = new_model

        test_predictions = new_model.predict(split_datasets["test"]["x"])
        test_cm = confusion_matrix(split_datasets["test"]["y"], test_predictions)
        pruned_scores[ccp_alpha] = {"Adjusted Precision": adj_precision(test_cm),
                                    "Adjusted Recall": adj_recall(test_cm),
                                    "Adjusted F1": adj_f1(adj_precision(test_cm), adj_recall(test_cm))}

    # best CCP alpha
    print(f"Best CCP Alpha: {best_ccp_alpha}")
    old_train_predictions = trained_tree.predict(split_datasets["train"]["x"])
    old_train_cm = confusion_matrix(split_datasets["train"]["y"], old_train_predictions)
    print("Old Tree: Training")
    print(f"Adjusted Precision: {adj_precision(old_train_cm)}")
    print(old_train_cm)

    old_test_predictions = trained_tree.predict(split_datasets["test"]["x"])
    old_test_cm = confusion_matrix(split_datasets["test"]["y"], old_test_predictions)
    print("Old Tree: Test")
    print(f"Adjusted Precision: {adj_precision(old_test_cm)}")
    print(old_test_cm)

    train_predictions = best_model.predict(split_datasets["train"]["x"])
    train_cm = confusion_matrix(split_datasets["train"]["y"], train_predictions)
    print("Pruned tree: Training")
    print(f"Adjusted Precision: {adj_precision(train_cm)}")
    print(train_cm)

    test_predictions = best_model.predict(split_datasets["test"]["x"])
    test_cm = confusion_matrix(split_datasets["test"]["y"], test_predictions)
    print("Pruned tree: Testing")
    print(f"Adjusted Precision: {adj_precision(test_cm)}")
    print(test_cm)

    graph_confusion_matrices(split_datasets["test"]["y"].IsStrongFlare.to_list(), old_test_predictions.tolist(), test_predictions.tolist(), results_folderpath, run_nickname, minutes_since_start)

    with open(out_filepath, 'a') as f:
        f.write(f"Minutes since start: {minutes_since_start}, Best CCP Alpha: {best_ccp_alpha}, best adjusted precision: {best_adj_precision}, adj_p diff: {best_adj_precision - adj_precision(old_test_cm)}\n")

    with open(os.path.join(models_dir, f"pruned_model_{minutes_since_start}_minutes_since_start.pkl"), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(scores_dir, f"pruned_model_scores_{minutes_since_start}_minutes_since_start.pkl"), 'wb') as f:
        pickle.dump(pruned_scores, f)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", type=str, help="Run nickname")
        parser.add_argument("-m", type=str, help="Model Type")
        args = parser.parse_args()
        with Pool(21) as p:
            p.starmap(ccp_alpha_prune, zip(repeat(args.n), repeat(args.m), range(-5, 16)))

    else:
        run_nickname = f"2025_03_06_multiclass_naive_adjusted_precision_dt"
        model_type = "Decision Tree"
        with Pool(4) as p:
            p.starmap(ccp_alpha_prune, zip(repeat(run_nickname), repeat(model_type), range(-5, 16)))
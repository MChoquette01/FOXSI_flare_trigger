import PySimpleGUI as psg
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tree_common as tc
import numpy as np


# GUI to tweak tree hyperparameters to visualize how classifications results/scores change

def get_metrics(t, train_x, train_y, test_x, test_y):

    t.fit(train_x.values, train_y.values)
    test_predictions = t.predict(test_x.values)

    scores = {"Training": {}, "Test": {}}

    # create confusion matrices (or rather, the related stats) for training and test sets
    test_y_reshaped = test_y.to_numpy().reshape((np.shape(test_y)[0],))
    # ConfusionMatrixDisplay.from_predictions(test_y, predictions, display_labels=["Small Flare", "Big Flare"])
    # plt.show()
    test_cm = confusion_matrix(test_y, test_predictions)
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()

    scores["Test"]["FPR"] = test_fp / (test_tp + test_fp)
    scores["Test"]["Precision"] = precision_score(test_y, test_predictions)
    scores["Test"]["Recall"] = recall_score(test_y, test_predictions)
    scores["Test"]["TSS"] = scores["Test"]["Recall"] - scores["Test"]["FPR"]
    scores["Test"]["F1"] = f1_score(test_y, test_predictions)

    train_predictions = t.predict(train_x.values)
    train_y_reshaped = train_y.to_numpy().reshape((np.shape(train_y)[0],))
    train_cm = confusion_matrix(train_y, train_predictions)
    train_tn, train_fp, train_fn, train_tp = train_cm.ravel()

    scores["Training"]["Precision"] = precision_score(train_y, train_predictions)
    scores["Training"]["Recall"] = recall_score(train_y, train_predictions)
    scores["Training"]["FPR"] = train_fp / (train_tp + train_fp)
    scores["Training"]["TSS"] = scores["Training"]["Recall"] - scores["Training"]["FPR"]
    scores["Training"]["F1"] = f1_score(train_y, train_predictions)

    return scores


psg.theme("Reddit")

training_metrics_str = ""
test_metrics_str = ""
training_cm_text = ""
test_cm_text = ""

layout = [[psg.FileBrowse("Choose Results Pickle", enable_events=True, key="Choose Results Pickle")],
          [psg.Frame("Updater", [[psg.Text(f"Training: {training_metrics_str}", key="Training Text")],
                                 [psg.Text(f"Test: {test_metrics_str}", key="Test Text")],
                                 [psg.Text("Minutes From Flare Start", key="Minutes From Start"), psg.Slider((-5, 3), orientation='horizontal', key='Minutes From Start Slider')],
                                 [psg.Button("Get Best For Timestamp", key="Get Best For Timestamp")],
                                 [psg.Text("Loss Method", key="Loss Label"), psg.Radio('Gini', group_id=1, key='gini', default=True), psg.Radio('Entropy', group_id=1, key='entropy')],
                                 [psg.Text("Max Depth", key="Max Depth Label"), psg.Slider((1, 30), orientation='horizontal', key="Max Depth Slider")],
                                 [psg.Text("Max Features", key="Max Depth Label"), psg.Slider((1, 18), orientation='horizontal', key="Max Features Slider")],
                                 [psg.Text("Min Samples Leaf", key="Min Samples Leaf Label"), psg.Slider((1, 30), orientation='horizontal', key="Min Samples Leaf Slider")],
                                 [psg.Text("Min Samples Split", key="Min Samples Split Label"), psg.Slider((1, 30), orientation='horizontal', key="Min Samples Split Slider")],
                                 [psg.Text("Min Weight Fraction Leaf", key="Min Weight Fraction Leaf Label"), psg.Slider((0.0, 0.5), resolution=0.1, orientation='horizontal', key="Min Weight Fraction Leaf Slider")],
                                 [psg.Text("CCP Alpha", key="CCP Alpha Label"), psg.Slider((0, 1.0), resolution=0.1, orientation='horizontal', key="CCP Alpha Slider")],
                                 [psg.Button("Update Tree", key="Update Tree")],
                                 [psg.Text("Training CM: ", key="Training CM Label"), psg.Text(training_cm_text, key='Training CM Text')],
                                 [psg.Text("Test CM: ", key="Test CM Label"), psg.Text(test_cm_text, key='Test CM Text')]], key="Update Frame", visible=False)]]

window = psg.Window(title="Pruner", layout=layout, margins=(100, 50))

# Create an event loop
while True:
    event, values = window.read()
    if event == "Choose Results Pickle":
        results_filepath = values["Choose Results Pickle"]
        with open(results_filepath, "rb") as f:
            results = pickle.load(f)
        t = tc.create_tree_from_df(results, minutes_since_start=10, max_depth_override=None, ccp_alpha=0.0)
        train_x, _, train_y, test_x, _, test_y = tc.get_train_and_test_data_from_pkl(minutes_since_start=10, peak_filtering_minutes=0)
        scores = get_metrics(t, train_x, train_y, test_x, test_y)
        training_metrics_str = f"Precision: {round(scores['Training']['Precision'], 3)}, " \
                               f"Recall: {round(scores['Training']['Recall'], 3)}," \
                               f"FPR: {round(scores['Training']['FPR'], 3)}," \
                               f"TSS: {round(scores['Training']['TSS'], 3)}," \
                               f"F1: {round(scores['Training']['F1'], 3)}"

        test_metrics_str = f"Precision: {round(scores['Test']['Precision'], 3)}, " \
                           f"Recall: {round(scores['Test']['Recall'], 3)}," \
                           f"FPR: {round(scores['Test']['FPR'], 3)}," \
                           f"TSS: {round(scores['Test']['TSS'], 3)}," \
                           f"F1: {round(scores['Test']['F1'], 3)}"
        window["Update Frame"].update(visible=True)
        window["Training Text"].update(value=training_metrics_str)
        window["Test Text"].update(value=test_metrics_str)
        relevant_row = results[results.minutes_since_start == str(10)]
        window["Max Depth Slider"].update(value=int(relevant_row.max_depth.iloc[0]))
        window["Max Features Slider"].update(value=int(relevant_row.max_features.iloc[0]))
        window["Min Samples Leaf Slider"].update(value=int(relevant_row.min_samples_leaf.iloc[0]))
        window["Min Samples Split Slider"].update(value=int(relevant_row.min_samples_split.iloc[0]))
        window["Min Weight Fraction Leaf Slider"].update(value=float(relevant_row.min_weight_fraction_leaf.iloc[0]))
        window["CCP Alpha Slider"].update(value=0)

        train_predictions = t.predict(train_x)
        train_cm = confusion_matrix(train_y, train_predictions)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_cm_str = f"TN: {train_tn}   FP: {train_fp}\nFN: {train_fn}   TP: {train_tp}"
        window["Training CM Text"].update(value=train_cm_str)

        test_predictions = t.predict(test_x)
        test_cm = confusion_matrix(test_y, test_predictions)
        test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
        test_cm_str = f"TN: {test_tn}   FP: {test_fp}\nFN: {test_fn}   TP: {test_tp}"
        window["Test CM Text"].update(value=test_cm_str)
        window.Refresh()
    elif event == "Update Tree":
        # find chosen split criteria
        for split_method in ['gini', 'entropy']:
            if values[split_method]:
                criterion = split_method
                break
        t = tc.create_tree(criterion,
                           int(values["Max Depth Slider"]),
                           int(values["Max Features Slider"]),
                           int(values["Min Samples Leaf Slider"]),
                           int(values["Min Samples Split Slider"]),
                           float(values["Min Weight Fraction Leaf Slider"]),
                           float(values["CCP Alpha Slider"]))

        train_x, _, train_y, test_x, _, test_y = tc.get_train_and_test_data_from_pkl(minutes_since_start=int(values['Minutes From Start Slider'] + 15), peak_filtering_minutes=0)
        scores = get_metrics(t, train_x, train_y, test_x, test_y)
        training_metrics_str = f"Precision: {round(scores['Training']['Precision'], 3)}, " \
                               f"Recall: {round(scores['Training']['Recall'], 3)}," \
                               f"FPR: {round(scores['Training']['FPR'], 3)}," \
                               f"TSS: {round(scores['Training']['TSS'], 3)}," \
                               f"F1: {round(scores['Training']['F1'], 3)}"

        test_metrics_str = f"Precision: {round(scores['Test']['Precision'], 3)}, " \
                           f"Recall: {round(scores['Test']['Recall'], 3)}," \
                           f"FPR: {round(scores['Test']['FPR'], 3)}," \
                           f"TSS: {round(scores['Test']['TSS'], 3)}," \
                           f"F1: {round(scores['Test']['F1'], 3)}"
        window["Training Text"].update(value=training_metrics_str)
        window["Test Text"].update(value=test_metrics_str)

        train_predictions = t.predict(train_x)
        train_cm = confusion_matrix(train_y, train_predictions)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_cm_str = f"TN: {train_tn}   FP: {train_fp}\nFN: {train_fn}   TP: {train_tp}"
        window["Training CM Text"].update(value=train_cm_str)

        test_predictions = t.predict(test_x)
        test_cm = confusion_matrix(test_y, test_predictions)
        test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
        test_cm_str = f"TN: {test_tn}   FP: {test_fp}\nFN: {test_fn}   TP: {test_tp}"
        window["Test CM Text"].update(value=test_cm_str)
        window.Refresh()
    elif event == "Get Best For Timestamp":
        relevant_row = results[results.minutes_since_start == str(int(values['Minutes From Start Slider']) + 15)]
        for split_criteria in ['gini', 'entropy']:
            if relevant_row.criterion.iloc[0] == split_criteria:
                window[split_criteria].update(True)
            else:
                window[split_criteria].update(False)
        window["Max Depth Slider"].update(value=int(relevant_row.max_depth.iloc[0]))
        window["Max Features Slider"].update(value=int(relevant_row.max_features.iloc[0]))
        window["Min Samples Leaf Slider"].update(value=int(relevant_row.min_samples_leaf.iloc[0]))
        window["Min Samples Split Slider"].update(value=int(relevant_row.min_samples_split.iloc[0]))
        window["Min Weight Fraction Leaf Slider"].update(value=float(relevant_row.min_weight_fraction_leaf.iloc[0]))
        window["CCP Alpha Slider"].update(value=0)
    elif event == "OK" or event == psg.WIN_CLOSED:
        break

window.close()

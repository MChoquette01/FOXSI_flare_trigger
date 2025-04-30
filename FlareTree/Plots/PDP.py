import pickle
import os
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

"""Plot Partial Dependency Plots for the two most important variables in each model
'What the heck is a PDP, you say? It shows the effect of one or two predictors on the predicted class!
See here: https://scikit-learn.org/stable/modules/partial_dependence.html"""

results_dir = "../MSI Results"
run_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"

if not os.path.exists(os.path.join("Saved Plots", "Partial Dependence Diagrams")):
    os.mkdir(os.path.join("Saved Plots", "Partial Dependence Diagrams"))

for minutes_since_start in range(-5, 16):
    feature_importance_filepath = os.path.join(results_dir, run_nickname, "Feature Importance", f"feature_importances_{minutes_since_start + 15}_minutes_since_flare_start.pkl")
    model_path = os.path.join(results_dir, run_nickname, "Trees", f"trained_{minutes_since_start}_minutes_since_start")
    split_datasets_path = os.path.join(results_dir, run_nickname, "Datasets", f"split_datasets{minutes_since_start}_minutes_since_start.pkl")

    targets = ["<C5", "C, >=C5", "M", "X"]

    #from GridSearch.py - order matters because this is how the model was trained!
    params = ["CurrentXRSA", "XRSA1MinuteDifference", "XRSA2MinuteDifference",
              "XRSA3MinuteDifference", "XRSA4MinuteDifference", "XRSA5MinuteDifference", "CurrentXRSB",
              "XRSB1MinuteDifference", "XRSB2MinuteDifference", "XRSB3MinuteDifference",
              "XRSB4MinuteDifference", "XRSB5MinuteDifference", "Temperature", "Temperature1MinuteDifference",
              "Temperature2MinuteDifference", "Temperature3MinuteDifference", "Temperature4MinuteDifference",
              "Temperature5MinuteDifference", "EmissionMeasure", "EmissionMeasure1MinuteDifference", "EmissionMeasure2MinuteDifference",
              "EmissionMeasure3MinuteDifference", "EmissionMeasure4MinuteDifference", "EmissionMeasure5MinuteDifference",
              "NaiveTemperature1MinuteDifference", "NaiveTemperature2MinuteDifference", "NaiveTemperature3MinuteDifference",
              "NaiveTemperature4MinuteDifference", "NaiveTemperature5MinuteDifference", "NaiveEmissionMeasure1MinuteDifference",
              "NaiveEmissionMeasure2MinuteDifference", "NaiveEmissionMeasure3MinuteDifference", "NaiveEmissionMeasure4MinuteDifference", "NaiveEmissionMeasure5MinuteDifference"]

    with open(feature_importance_filepath, 'rb') as f:
        feature_importances = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(split_datasets_path, 'rb') as f:
        split_datasets = pickle.load(f)

    features = [params.index(feature_importances[-1][0]), params.index(feature_importances[-2][0]), (params.index(feature_importances[-1][0]), params.index(feature_importances[-2][0]))]
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    row = 0
    col = 0

    for target_idx in range(4):
        ax[row, col].set_title(targets[target_idx])
        try:
            PartialDependenceDisplay.from_estimator(model, split_datasets['train']['x'], features, target=target_idx, ax=ax[row, col])
        except ValueError:
            continue
        col = (target_idx + 1) % 2
        row = 1 if target_idx + 1 >= 2 else 0
    plt.suptitle(f"Partial Dependence Plots: {minutes_since_start} Minutes Since Flare Start", fontsize=22)
    # plt.subplots_adjust(left=0.1)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join("Saved Plots", "Partial Dependence Diagrams", f"PDD_{minutes_since_start}_minutes_since_start.png"))
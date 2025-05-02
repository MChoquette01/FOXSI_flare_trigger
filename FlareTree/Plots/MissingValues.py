import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

"""Plot frequency of missing values in a dataset. Broken out by temperature/emission measure and XRSA/B flux."""

parsed_flares_dir = r"peak_threshold_minutes_-10000_multiclass_naive"

lower_bound_cutoff_percent = 1

metric_values = defaultdict(list)
for time_minutes in range(10, 31):
    with open(os.path.join("../Parsed Flares", parsed_flares_dir, f"{time_minutes}_minutes_since_start.pkl"), "rb") as f:
        parsed_flares = pickle.load(f)

        for variable in list(parsed_flares.columns):
            if variable == "FlareID" or variable == "IsStrongFlare" or variable == "MinutesToPeak" or variable == "FlareClass":
                continue
            metric_values[variable].append(parsed_flares[variable].isna().sum() / parsed_flares.shape[0] * 100)

# plot temp and EM
for metric_type in ["Temperature", "EmissionMeasure"]:
    plt.figure(figsize=(16, 9))
    xs = [int(x) for x in range(10, 31)]
    markers = [".", "^", "s", "x", "*", "+"]
    marker_idx = 0
    for metric, values in metric_values.items():
        if (metric_type in metric) and ("Naive" not in metric):
            plt.plot([x - 15 for x in xs], values, label=metric, marker=markers[marker_idx])
            marker_idx += 1
    plt.legend(fontsize='xx-large', bbox_to_anchor=(1.05, 0.95))
    plt.title(f"Missing Values ({metric_type})", fontsize=24)
    plt.xlabel("Minutes Since Flare Start", fontsize=22)
    plt.ylabel("Percent of Missing Values", fontsize=22)
    plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "nan_frequency.png"))
    # plt.show()
    if not os.path.exists("Saved Plots"):
        os.mkdir("Saved Plots")
    # plt.savefig(os.path.join("Saved Plots", "TempAmdEMMissingValues.png"))

# Plot XRSA Flux
for flux_type in ["XRSA", "XRSB"]:
    plt.figure(figsize=(16, 9))
    xs = [int(x) for x in range(10, 31)]
    markers = [".", "^", "s", "x", "*", "+"]
    marker_idx = 0
    for metric, values in metric_values.items():
        if flux_type in metric:
            plt.plot([x - 15 for x in xs], values, label=metric, marker=markers[marker_idx])
            marker_idx += 1
    plt.legend(fontsize='xx-large', bbox_to_anchor=(1.05, 0.95))
    plt.title(f"Missing Values ({flux_type} Flux)", fontsize=24)
    plt.xlabel("Minutes Since Flare Start", fontsize=22)
    plt.ylabel("Percent of Missing Values", fontsize=22)
    plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    if not os.path.exists("Saved Plots"):
        os.mkdir("Saved Plots")
    # plt.savefig(os.path.join("Saved Plots", "FluxMissingValues.png"))
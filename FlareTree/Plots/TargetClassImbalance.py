import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import sys
sys.path.insert(0, "../")
import tree_common as tc
from collections import defaultdict

"""Plot the (in)balance in the classification TARGETs in a dataset"""

parsed_flares_dir = r"peak_threshold_minutes_-10000_multiclass_naive"

flare_class_totals = defaultdict(list)
for time_minutes in range(10, 31):
    with open(os.path.join("../Parsed Flares", parsed_flares_dir, f"{time_minutes}_minutes_since_start.pkl"), "rb") as f:
        parsed_flares = pickle.load(f)

    parsed_flares = tc.assign_is_strong_flare_column(parsed_flares)

    for target_class in [0, 1, 2, 3]:
        subset = parsed_flares[parsed_flares.IsStrongFlare == np.float64(target_class)]
        flare_class_totals[target_class].append(subset.shape[0])


# plot
plt.figure(figsize=(16, 9))
labels = ["<C5", "C, >=C5", "M", "X"]
xs = [int(x) for x in range(10, 31)]
for idx in range(-5, 16):
    bottom = 0
    for class_id, color in zip(range(4), ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        if idx != 1:
            plt.bar(idx, flare_class_totals[class_id][idx], bottom=bottom, color=color)
        else:
            plt.bar(idx, flare_class_totals[class_id][idx], bottom=bottom, color=color, label=labels[class_id])
        bottom += flare_class_totals[class_id][idx]

plt.legend(fontsize="xx-large")
plt.title("Distribution of Maximum XRSB Flux 8-14 Minutes Out", fontsize=24)
plt.xlabel("Minutes Since Flare Start", fontsize=22)
plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Count of Flares", fontsize=22)
# plt.savefig(os.path.join(out_dir, "nan_frequency.png"))
# plt.show()
if not os.path.exists("Saved Plots"):
    os.mkdir("Saved Plots")
plt.savefig(os.path.join("Saved Plots", "TargetClassImbalance.png"))
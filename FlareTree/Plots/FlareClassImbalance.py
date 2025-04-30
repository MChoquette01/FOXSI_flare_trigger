import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.insert(0, "../")
import tree_common as tc
from collections import defaultdict

"""Graph the (im)balance of flare magnitude in a dataset"""

parsed_flares_dir = r"peak_threshold_minutes_-10000_multiclass_naive"

flare_class_totals = defaultdict(list)
for time_minutes in range(10, 31):
    with open(os.path.join("../Parsed Flares", parsed_flares_dir, f"{time_minutes}_minutes_since_start.pkl"), "rb") as f:
        parsed_flares = pickle.load(f)

    parsed_flares = tc.assign_is_strong_flare_column(parsed_flares)

    for flare_class in ["B", "C, <C5", "C, >=C5", "M", "X"]:
        if not "C" in flare_class:
            relevant_indicies = parsed_flares.index[parsed_flares.FlareClass.str.startswith(flare_class)].tolist()
        else:
            if flare_class == "C, <C5":
                relevant_indicies = parsed_flares.index[(parsed_flares.FlareClass.str.startswith("C")) & (parsed_flares["IsFlareClass>=C5"] == False)].tolist()
            elif flare_class == "C, >=C5":
                relevant_indicies = parsed_flares.index[(parsed_flares.FlareClass.str.startswith("C")) & (parsed_flares["IsFlareClass>=C5"] == True)].tolist()

        subset = parsed_flares.iloc[relevant_indicies]
        flare_class_totals[flare_class].append(subset.shape[0])


# plot
plt.figure(figsize=(16, 9))
xs = [int(x) for x in range(10, 31)]
for minutes_since_start in xs:
    bottom = 0
    for flare_class, color in zip(["B", "C, <C5", "C, >=C5", "M", "X"], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        if minutes_since_start != 10:
            plt.bar(minutes_since_start - 15, flare_class_totals[flare_class][minutes_since_start - 10], color=color, bottom=bottom)
        else:
            plt.bar(minutes_since_start - 15, flare_class_totals[flare_class][minutes_since_start - 10], color=color, bottom=bottom, label=flare_class)
        bottom += flare_class_totals[flare_class][minutes_since_start - 10]
plt.legend(fontsize='xx-large')
plt.title("Dataset GOES Flare Class Distribution", fontsize=24)
plt.xlabel("Minutes Since Flare Start", fontsize=22)
plt.xticks(ticks=[x for x in range(-5, 16)], labels=[x for x in range(-5, 16)], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Count of Flares", fontsize=22)
# plt.show()
if not os.path.exists("Saved Plots"):
    os.mkdir("Saved Plots")
plt.savefig(os.path.join("Saved Plots", "FlareClassImbalance.png"))
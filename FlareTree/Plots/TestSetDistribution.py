import matplotlib.pyplot as plt
import os
import pickle
import pandas
from collections import Counter

"""Plot GOES classes for the test set"""

output_dir = r"../MSI Results"
run_nickname = "2025_03_21_multiclass_naive_adjusted_precision_gbdt"

# minutes don't matter since we're using GOES class here...use 0
datasets_filepath = os.path.join(output_dir, run_nickname, "Datasets", "split_datasets0_minutes_since_start.pkl")
with open(datasets_filepath, 'rb') as f:
    split_datasets = pickle.load(f)

flare_classes = split_datasets["test"]["additional_data"].FlareClass.values
flare_mags = sorted([x[:1] for x in flare_classes])
letter_counts = Counter(flare_mags)
df = pandas.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')
plt.show()

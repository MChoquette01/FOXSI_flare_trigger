import matplotlib.pyplot as plt
import os
import pickle
import pandas
from collections import Counter

"""Examine the magnitude distribution of the ensemble test set"""

output_dir = r"C:../MSI Results\2025_03_21_multiclass_naive_adjusted_precision_gbdt"

# minutes don't matter since we're using GOES class here...use 0
datasets_filepath = os.path.join(output_dir, "Datasets", "split_datasets0_minutes_since_start.pkl")
with open(datasets_filepath, 'rb') as f:
    split_datasets = pickle.load(f)

flare_classes = split_datasets["temporal_test"]["additional_data"].FlareClass.values
flare_mags = sorted([x[:2] for x in flare_classes])  # first two characters only
letter_counts = Counter(flare_mags)
df = pandas.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')
plt.show()

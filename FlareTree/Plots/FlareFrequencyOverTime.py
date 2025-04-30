import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import os
from dateutil.relativedelta import relativedelta
import sys
sys.path.insert(0, "../")
import tree_common as tc

"""Plot flare magnitude and frequency over time, binned by month"""

result = []
today = datetime.strptime("2024 2", "%Y %m")

flare_timestamps = {"B": {}, "C, <C5": {}, "C, >=C5": {}, "M": {}, "X": {}}
for flare_class in list(flare_timestamps.keys()):
    flare_timestamps_for_class = flare_timestamps[flare_class]
    current = datetime.strptime("2017 2", "%Y %m")
    while current <= today:
        result.append(current)
        flare_timestamps_for_class[current] = 0
        current += relativedelta(months=1)

def assign_label_class(flare_class):

    if flare_class[0] == "B":
        return "B"
    elif flare_class[0] == "C":
        if float(flare_class[1:]) < 5.0:
            return "C, <C5"
        else:
            return "C, >=C5"
    elif flare_class[0] == "M":
        return "M"
    elif flare_class[0] == "X":
        return "X"


client, flares_table = tc.connect_to_flares_db()
filter={'FlareID': {'$regex': '_0$'}}
project={'FlareID': 1, 'FlareClass': 1}
cursor = client['Flares']['NaiveFlares'].find(filter=filter, projection=project)
for record in cursor:
    year = record["FlareID"][:4]
    month = record["FlareID"][4:6]
    flare_time = datetime.strptime(f"{year} {month}", "%Y %m")
    flare_timestamps[assign_label_class(record['FlareClass'])][flare_time] += 1

plt.figure(figsize=(16, 9))
xs = [x for x in range(len(flare_timestamps["B"]))]
ys = list(flare_timestamps.values())
for idx, timestamp in enumerate(flare_timestamps["B"]):
    bottom = 0
    for flare_class, color in zip(["B", "C, <C5", "C, >=C5", "M", "X"], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        if idx != 0:
            plt.bar(idx, flare_timestamps[flare_class][timestamp], color=color, bottom=bottom)
        else:
            plt.bar(idx, flare_timestamps[flare_class][timestamp], color=color, bottom=bottom, label=flare_class)
        bottom += flare_timestamps[flare_class][timestamp]
labels = [datetime.strftime(x, "%m-%Y") for x in list(flare_timestamps["B"].keys())]
plt.xticks(ticks=xs[::4], labels=labels[::4], rotation=75, fontsize=20)
plt.yticks(fontsize=20)
plt.title("Temporal Magnitude Distribution of Solar Flares Analyzed", fontsize=24)
plt.ylabel("Flare Count", fontsize=22)
plt.xlabel("Date (MM-YYYY)", fontsize=22)
plt.legend(fontsize='xx-large')
plt.tight_layout()
# plt.show()
if not os.path.exists("Saved Plots"):
    os.mkdir("Saved Plots")
plt.savefig(os.path.join("Saved Plots", "FlareFrequencyOverTime.png"))
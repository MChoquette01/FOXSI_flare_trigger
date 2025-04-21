import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.insert(0, "../")
import tree_common as tc

"""Plot an X-ray light curve!"""

# get random flare ID
# client, flares_table = tc.connect_to_flares_db()
# for record in flares_table.aggregate([{'$sample': {'size': 1 } }]):
#     flare_id = record["FlareID"].split("_")[0]

# or plot your faovrite flare instead!
flare_id = "202112280729"
datums_to_graph = ["CurrentXRSB"]  # what datums to graph...verbiage should match keys in MongoDB


datum_name_map = {"CurrentXRSA": "XRSA",
                  "CurrentXRSB": "XRSB",
                  "XRSA1MinuteDifference": "XRSA 1 Minute Difference",
                  "XRSA2MinuteDifference": "XRSA 2 Minute Difference",
                  "XRSA3MinuteDifference": "XRSA 3 Minute Difference",
                  "XRSA4MinuteDifference": "XRSA 4 Minute Difference",
                  "XRSA5MinuteDifference": "XRSA 5 Minute Difference",
                  "XRSB1MinuteDifference": "XRSB 1 Minute Difference",
                  "XRSB2MinuteDifference": "XRSB 2 Minute Difference",
                  "XRSB3MinuteDifference": "XRSB 3 Minute Difference",
                  "XRSB4MinuteDifference": "XRSB 4 Minute Difference",
                  "XRSB5MinuteDifference": "XRSB 5 Minute Difference",
                  "Temperature": "Temperature",
                  "EmissionMeasure": "Emission Measure"
                  }

datums = defaultdict(list)

client, flares_table = tc.connect_to_flares_db()
filter={'FlareID': {'$regex': f'^{flare_id}'}}
projection = {}
for datum in datums_to_graph:
    projection[datum] = 1
sort = list({'MinutesToPeak': -1}.items())
cursor = flares_table.find(filter=filter, projection=projection, sort=sort)
for record in cursor:
    for datum in datums_to_graph:
        datums[datum].append(record[datum])
foo = 2

for datum_name, values in datums.items():
    plt.plot([x - 15 for x in range(len(values))], values, label=datum_name_map[datum_name])
plt.ylabel("W/m^2")
plt.title(f"Flare ID: {flare_id}")
plt.legend()
plt.show()
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import os
import sys
sys.path.insert(0, "../")
import tree_common as tc

"""Get X-ray light curve data for some flares graphed at early timesteps and the curve, to show the two are not usually correlated"""

def get_data_for_hardcoded_flares():

    flare_ids_to_graph = ["202207021811", "202105290346", "202203281703", "202110090619", "202204200341"]

    data_to_graph = {}
    client, flares_table = tc.connect_to_flares_db()
    for flare_id in flare_ids_to_graph:

        # now get all XRSB datums for this flare
        filter = {'FlareID': {'$regex': f'^{flare_id}'}}
        projection = {'CurrentXRSB': 1}
        sort = list({'MinutesToPeak': -1}.items())  # sort chronologically
        cursor = flares_table.find(filter=filter, projection=projection, sort=sort)
        data_to_graph[flare_id] = []
        for idx, record in enumerate(cursor):
            data_to_graph[flare_id].append(record["CurrentXRSB"])
            if idx >= 70:
                break

    return data_to_graph


def get_random_data():

    flare_classes_query = ["B", "C1", "C9", "M", "X"]

    data_to_graph = {}
    client, flares_table = tc.connect_to_flares_db()
    for flare_class in flare_classes_query:
        filter = {'FlareClass': {'$regex': f'^{flare_class}'}, 'FlareID': {'$regex': '_0$'}}
        # result = flares_table.find(filter=filter, projection=project)
        number_of_flares_in_query = flares_table.count_documents(filter=filter)
        cursor = flares_table.find(filter=filter)
        random_flare_index = random.randint(0, number_of_flares_in_query)
        for idx, record in enumerate(cursor):
            if idx < random_flare_index:
                continue
            flare_id = record["FlareID"]
            break

        # now get all XRSB datums for this flare
        filter = {'FlareID': {'$regex': f'^{flare_id.split("_")[0]}'}}
        projection = {'CurrentXRSB': 1}
        sort = list({'MinutesToPeak': -1}.items())
        cursor = flares_table.find(filter=filter, projection=projection, sort=sort)
        data_to_graph[flare_id] = []
        for idx, record in enumerate(cursor):
            data_to_graph[flare_id].append(record["CurrentXRSB"])
            if idx >= 70:
                break

    return data_to_graph


def graph(data_to_graph):

    flare_classes = ["B", "Low C", "High C", "M", "X"]
    markers = [".", "^", "s", "x", "*", "+"]

    plt.figure(figsize=(16, 9))
    for (flare_id, xrsb), flare_class, marker in zip(data_to_graph.items(), flare_classes, markers):
        label = f"{flare_id[:4]}-{flare_id[4:6]}-{flare_id[6:8]}, {flare_class} Class"
        plt.plot([x - 15 for x in range(len(xrsb))], xrsb, label=label, marker=marker)
    max_tick_mark = max([len(x) for x in list(data_to_graph.values())])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.xticks(ticks=[x for x in range(-15, max_tick_mark)][::5], labels=[x for x in range(-15, max_tick_mark)][::5], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("XRSB Flux ($W m^{-2}$)", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=22)
    plt.title("Full XRSB Light Curves", fontsize=24)
    plt.legend(fontsize='xx-large')
    # plt.show()
    if not os.path.exists("Saved Plots"):
        os.mkdir("Saved Plots")
    plt.savefig(os.path.join("Saved Plots", "SampleLightCurvesFull.png"))

    # early flare inset
    plt.clf()
    plt.figure(figsize=(16, 9))
    for (flare_id, xrsb), flare_class, marker in zip(data_to_graph.items(), flare_classes, markers):
        label = f"{flare_id[:4]}-{flare_id[4:6]}-{flare_id[6:8]}, {flare_class} Class"
        plt.plot([x - 15 for x in range(len(xrsb))][10:22], xrsb[10:22], label=label, marker=marker)
    max_tick_mark = 22
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.xticks(ticks=[x for x in range(-10, max_tick_mark)][::5], labels=[x for x in range(-10, max_tick_mark)][::5], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("XRSB Flux ($W m^{-2}$)", fontsize=22)
    plt.xlabel("Minutes Since Flare Start", fontsize=22)
    plt.title("Pre-Flare and Early Flare XRSB Light Curves", fontsize=24)
    plt.legend(fontsize='xx-large')
    # plt.show()
    if not os.path.exists("Saved Plots"):
        os.mkdir("Saved Plots")
    plt.savefig(os.path.join("Saved Plots", "SampleLightCurvesSubset.png"))
#

# graph(get_random_data())
graph(get_data_for_hardcoded_flares())
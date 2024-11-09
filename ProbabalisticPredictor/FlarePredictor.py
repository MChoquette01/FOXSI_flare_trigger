import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient


class PercentThreshold:
    """Defines similar flares as those with a variable with a given percentage of the base flare"""
    def __init__(self, xrsb_percent_tolerance=0.1, xrsb_difference_tolerance=0.1):

        self.threshold_type = "Percent"
        self.xrsb_percent_tolerance = xrsb_percent_tolerance
        self.xrsb_difference_tolerance = xrsb_difference_tolerance
        self.thresholds = {}

    def calculate_thresholds(self, xrsb_flux, xrsb_three_minute_difference):

        self.thresholds["current_xrsb_lower_bound"] = xrsb_flux * (1 - self.xrsb_percent_tolerance)
        self.thresholds["current_xrsb_upper_bound"] = xrsb_flux * (1 + self.xrsb_percent_tolerance)

        # xrsa_three_minute_difference_lower_bound = xrsa_three_minute_difference * (1 - xrsa_three_minute_difference)
        # xrsa_three_minute_difference_upper_bound = xrsa_three_minute_difference * (1 + xrsa_three_minute_difference)

        self.thresholds["xrsb_three_minute_difference_lower_bound"] = xrsb_three_minute_difference * (1 - self.xrsb_difference_tolerance)
        self.thresholds["xrsb_three_minute_difference_upper_bound"] = xrsb_three_minute_difference * (1 + self.xrsb_difference_tolerance)


class LinearBinThreshold:
    """Creates linear bins of size <step> from 0 upwards. Similar flares to the base flare are defined as those with
    a value in the same bin"""

    def __init__(self, xrsb_flux_step=1*10**-8, xrsb_difference_step=1*10**-8):

        self.threshold_type = "Linear Bin"
        self.xrsb_flux_step = xrsb_flux_step
        self.xrsb_difference_step = xrsb_difference_step
        self.thresholds = {}

    def calculate_thresholds(self, xrsb_flux, xrsb_three_minute_difference):

        self.thresholds["current_xrsb_lower_bound"] = xrsb_flux - xrsb_flux % self.xrsb_flux_step
        self.thresholds["current_xrsb_upper_bound"] = self.thresholds["current_xrsb_lower_bound"] + self.xrsb_flux_step

        # self.thresholds["xrsa_three_minute_difference_lower_bound"] = xrsb_three_minute_difference - round(xrsb_three_minute_difference % self.step, 2)
        # self.thresholds["xrsa_three_minute_difference_upper_bound"] = self.thresholds["xrsa_three_minute_difference_lower_bound"] + self.step

        self.thresholds["xrsb_three_minute_difference_lower_bound"] = xrsb_three_minute_difference - xrsb_three_minute_difference % self.xrsb_difference_step
        self.thresholds["xrsb_three_minute_difference_upper_bound"] = self.thresholds["xrsb_three_minute_difference_lower_bound"] + self.xrsb_difference_step


class LinearMedianThreshold:
    """Creates a region of size <step> / 2 above adn below a value, and finds defines similar flares as those
    with values in that region"""

    def __init__(self, xrsb_flux_step=1*10**-8, xrsb_difference_step=1*10**-8):

        self.threshold_type = "Linear Median"
        self.xrsb_flux_step = xrsb_flux_step
        self.xrsb_difference_step = xrsb_difference_step
        self.thresholds = {}

    def calculate_thresholds(self, xrsb_flux, xrsb_three_minute_difference):

        self.thresholds["current_xrsb_lower_bound"] = xrsb_flux - self.xrsb_flux_step / 2
        self.thresholds["current_xrsb_upper_bound"] = xrsb_flux + self.xrsb_flux_step / 2

        self.thresholds["xrsb_three_minute_difference_lower_bound"] = xrsb_three_minute_difference - self.xrsb_difference_step / 2
        self.thresholds["xrsb_three_minute_difference_upper_bound"] = xrsb_three_minute_difference + self.xrsb_difference_step / 2


class LogThreshold:
    """Creates log10 bins of size <step> from 0 upwards. Similar flares to the base flare are defined as those with
    a value in the same bin"""

    def __init__(self, xrsb_flux_step=1*10**-8, xrsb_difference_step=1*10**-8):

        self.threshold_type = "Log"
        self.xrsb_flux_step = xrsb_flux_step
        self.xrsb_difference_step = xrsb_difference_step
        self.thresholds = {}

    def calculate_thresholds(self, xrsb_flux, xrsb_three_minute_difference):

        self.thresholds["current_xrsb_lower_bound"] = 10 ** (np.log10(xrsb_flux) - np.log10(xrsb_flux) % self.xrsb_flux_step)
        self.thresholds["current_xrsb_upper_bound"] = self.thresholds["current_xrsb_lower_bound"] + self.xrsb_flux_step

        # self.thresholds["xrsa_three_minute_difference_lower_bound"] = 10 ** (np.log10(xrsa_three_minute_difference) - np.log10(xrsa_three_minute_difference) % self.step)
        # self.thresholds["xrsa_three_minute_difference_upper_bound"] = self.thresholds["xrsa_three_minute_difference_lower_bound"] + self.step

        self.thresholds["xrsb_three_minute_difference_lower_bound"] = 10 ** (np.log10(xrsb_three_minute_difference) - np.log10(xrsb_three_minute_difference) % self.xrsb_difference_step)
        self.thresholds["xrsb_three_minute_difference_upper_bound"] = self.thresholds["xrsb_three_minute_difference_lower_bound"] + self.xrsb_difference_step


def get_unique_flare_count(flares_table):
    """Returns the count of unique flare events"""

    return flares_table.count_documents({"FlareID": {'$regex': r"(.*?)_0"}})


def get_timestamp_count_for_flare(flares_table, flare_id):
    """Returns the count of timestamped events for a given flare"""

    return flares_table.count_documents({"FlareID": {'$regex': rf"{flare_id}_(.*?)"}})


def get_flare_id(flares_table, flare_index):
    """Returns the id of a unique flare given its index in the DB"""

    for record in flares_table.find().skip(flare_index).limit(1):
        return record["FlareID"].split("_")[0]


def get_unique_flare_ids(flares_table):
    """Returns a list of unique flare IDs (ignoring the timestamp part - just the real IDs)"""

    flare_ids = []
    first_flare_entries = flares_table.find({"FlareID": {'$regex': r"(.*?)_0"}})
    for record in first_flare_entries:
        flare_ids.append(record["FlareID"].split("_")[0])

    return flare_ids


def find_base_flare(flares_table, flare_ids, flare_index, flare_timestamp):
    """Given index <flare_index> and the timestamp of that flare, returns info about that flare)"""

    first_flare_entries = flares_table.find({"FlareID": f"{flare_ids[flare_index]}_{flare_timestamp}"}).limit(1)

    base_flare = {}
    for first_record in first_flare_entries:
        base_flare["flare_id"] = first_record["FlareID"].split("_")[0]
        base_flare["xrsb_flux"] = first_record["CurrentXRSB"]
        base_flare["xrsb_three_minute_difference"] = first_record["XRSB3MinuteDifference"]
        base_flare["xrsb_remaining"] = eval(first_record["XRSBRemaining"])
        base_flare["flare_class"] = first_record["FlareClass"]
        base_flare["peak_timestamp"] = first_record["MinutesToPeak"]
        first_flare_entry = flares_table.find({"FlareID": f"{flare_ids[flare_index]}_0"}).limit(1)
        for record in first_flare_entry:
            base_flare["full_xrsb"] = eval(record["XRSBRemaining"])
        # current_xrsa = first_record["CurrentXRSA"]
        # xrsa_three_minute_difference = first_record["XRSA3MinuteDifference"]
        # xrsa_remaining = eval(first_record["XRSARemaining"])

    return base_flare


def find_similar_flares(base_flare_id, flares_table, thresholds):
    """Returns flares which are 'similar' to a pre-defined base flare"""

    find_similar_flare_query = {
        "CurrentXRSB": {"$lte": thresholds["current_xrsb_upper_bound"], "$gte": thresholds["current_xrsb_lower_bound"]},
        "XRSB3MinuteDifference": {"$lte": thresholds["xrsb_three_minute_difference_upper_bound"],
                                  "$gte": thresholds["xrsb_three_minute_difference_lower_bound"]},
        "MinutesToPeak": {"$gt": 0}}  # flares which haven't yet peaked

    similar_flares = []
    similar_flare_ids = []
    records = flares_table.find(find_similar_flare_query)
    for x in records:
        true_flare_id = x["FlareID"].split("_")[0]
        if true_flare_id not in similar_flare_ids and true_flare_id != base_flare_id:
            similar_flare_ids.append(true_flare_id)
            similar_flares.append(x)

    return similar_flares


def analyze_similar_flare_classes(similar_flares):
    """Bins similar_flares by GOES class. Returns dictionary of counts and a text annotation for graphing"""

    flare_classes = ["Unknown", "A", "B", "C < 5", "C >= 5", "M", "X"]
    flare_class_counts = dict(zip(flare_classes, [0] * len(flare_classes)))
    textstr = ""
    if similar_flares:
        similar_classes = []
        for flare in similar_flares:
            # TODO: Not reported, find peak flux manually
            if flare["FlareClass"] != "":
                flare_class = flare["FlareClass"][0]
                if flare_class != "C":
                    similar_classes.append(flare_class)
                else:
                    if 5.0 <= float(flare["FlareClass"][1:]):
                        similar_classes.append("C >= 5")
                    else:
                        similar_classes.append("C < 5")
            else:
                similar_classes.append("Unknown")
        for flare_class in flare_classes:
            flare_class_counts[flare_class] = similar_classes.count(flare_class)
            this_flare_class_percentage = round((similar_classes.count(flare_class) / len(similar_classes)) * 100, 2)
            # print(f"{flare_class}: {this_flare_class_percentage}%")
            textstr += f"{flare_class}: {this_flare_class_percentage}%\n"

        return flare_class_counts, textstr
    return flare_class_counts, textstr


def graph(base_flare, similar_flares, flare_timestamp, threshold_object, class_text):
    """Graph a base flare and flares similar to it"""

    graphed_any = False
    for similar_flare in similar_flares:
        pred_xrsb_remaining = eval(similar_flare["XRSBRemaining"])
        if min(pred_xrsb_remaining) < 0:
            print("Found a flare with negative flux! Skipping...")
            continue
        xs = np.arange(flare_timestamp, flare_timestamp + len(pred_xrsb_remaining))
        if graphed_any:
            plt.plot(xs, pred_xrsb_remaining, linestyle="--", color="green")
        else:
            plt.plot(xs, pred_xrsb_remaining, linestyle="--", color="green", label="Similar Flare")
            graphed_any = True
    plt.plot(base_flare["full_xrsb"], color="orange", label="Observed")
    plt.yscale('log')
    plt.axvline(flare_timestamp, linestyle="--", color="black", label="Prediction Time")

    # Draw standard GOES classes
    y_min, y_max = plt.gca().get_ylim()
    x_min, x_max = plt.gca().get_xlim()
    if y_min <= 10 ** -7 <= y_max:
        plt.axhline(10 ** -7, linestyle="--", color="blue")
        plt.text(x_min, 10 ** -7, 'B Class')
    if y_min <= 10 ** -6 <= y_max:
        plt.axhline(10 ** -6, linestyle="--", color="blue")
        plt.text(x_min, 10 ** -6, 'C Class')
    if y_min <= 10 ** -5 <= y_max:
        plt.axhline(10 ** -5, linestyle="--", color="blue")
        plt.text(x_min, 10 ** -5, 'M Class')
    if 10 ** -4 <= y_max:
        plt.axhline(10 ** -4, linestyle="--", color="blue")
        plt.text(x_min, 10 ** -4, 'X Class')

    # Add class percentages as text annotation
    props = dict(boxstyle='round', facecolor='wheat')
    plt.text(plt.gca().get_xlim()[1] - 27, plt.gca().get_ylim()[1], class_text, fontsize=14, verticalalignment='top', bbox=props)
    plt.legend(loc="center right")
    subtitle_text = f"Similarity: Current XRSB Flux: ({'{:.2E}'.format(threshold_object.thresholds['current_xrsb_lower_bound'], 2)} - {'{:.2E}'.format(threshold_object.thresholds['current_xrsb_upper_bound'], 2)})" \
                    f"    XRSB 3 Min. Diff: ({'{:.2E}'.format(threshold_object.thresholds['xrsb_three_minute_difference_lower_bound'], 2)} - {'{:.2E}'.format(threshold_object.thresholds['xrsb_three_minute_difference_upper_bound'], 2)})"
    plt.title(f"{threshold_object.threshold_type} Similarity: Flare ID {base_flare['flare_id']} ({base_flare['flare_class']})\n{subtitle_text}")
    plt.xlabel("Time (Minutes from start of clipped base flare)")
    plt.ylabel("XRSB Flux (W/m^2)")
    plt.show()


if __name__ == "__main__":

    # these are zero-indexed
    flare_index = 5                  # unique flare in the DB
    flare_timestamp = 5             # Nth timestamp of that flare to predict from

    client = MongoClient("mongodb://localhost:27017/")
    flares_db = client["Flares"]
    flares_table = flares_db["Flares"]

    flare_ids = get_unique_flare_ids(flares_table)
    lt = LinearBinThreshold(xrsb_flux_step=1*10**-8, xrsb_difference_step=1*10**-8)
    base_flare = find_base_flare(flares_table, flare_ids, flare_index, flare_timestamp)
    lt.calculate_thresholds(base_flare["xrsb_flux"],
                            base_flare["xrsb_three_minute_difference"])
    similar_flares = find_similar_flares(base_flare["flare_id"], flares_table, lt.thresholds)
    similar_flare_classes, class_text = analyze_similar_flare_classes(similar_flares)
    graph(base_flare, similar_flares, flare_timestamp, lt, class_text)

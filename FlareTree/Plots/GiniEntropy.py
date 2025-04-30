import matplotlib.pyplot as plt
import math
import numpy
import os

"""Plot a dandy graph comparing the Gini and Entropy impurity functions, for the tree fans out there"""


def entropy(positive_class_proportion):

    negative_class_proportion = 1 - positive_class_proportion
    entropy = - positive_class_proportion * math.log(positive_class_proportion, 2) - (negative_class_proportion * math.log(negative_class_proportion, 2))
    return entropy


def gini(positive_class_proportion):

    negative_class_proportion = 1 - positive_class_proportion
    gini = 1 - (positive_class_proportion)**2 - (negative_class_proportion)**2
    return gini

xs = numpy.arange(0, 1, 1 / 1000)
entropy_ys = [entropy(x) for x in xs[1:]]
gini_ys = [gini(x) for x in xs[1:]]


plt.figure(figsize=(16, 9))
plt.plot(xs[1:], entropy_ys, label="Entropy")
plt.plot(xs[1:], gini_ys, label="Gini Impurity")
plt.xticks(ticks=[x / 100 for x in range(101)][::10], labels=[x / 100 for x in range(101)[::10]], fontsize=18)
plt.yticks(ticks=[x / 100 for x in range(101)][::10], labels=[x / 100 for x in range(101)][::10], fontsize=18)
plt.xlabel("Proportion of Records in Class 1", fontsize=22)
plt.ylabel("Impurity", fontsize=22)
plt.title("Binary Classification Gini and Entropy", fontsize=24)
plt.legend(fontsize='xx-large')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join("Saved Plots", "Gini and Entropy.png"))
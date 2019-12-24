import sys
import os

import tensorflow
import numpy as np
from setuptools.sandbox import save_argv
from sympy import harmonic
from tensorflow import keras
from brian2 import *
from random import random
from matplotlib.pyplot import *

from network import generate_network

def save_heatmap(
        array,
        file_path,
        plot_title ="Untitled"
):
    fig, ax = plt.subplots()
    im = ax.imshow(array, aspect="auto")

    for i in range(len(array)):
        for j in range(len(array[i])):
            text = ax.text(j, i, array[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(plot_title)
    fig.tight_layout()
    plt.savefig(file_path)

# Output directory control
# output_path = os.path.join(os.environ["OUTPUT_PATH"])



output_path = "../output/test.out"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

data_file = open("../data/binary.data", "r")
data_lines = data_file.readlines()
data = list(map(lambda x: list(map(float, x.split(","))), data_lines))

temp0_mat = np.zeros(shape=(30, 20))
temp1_mat = np.zeros(shape=(30, 20))
count = 0
result = generate_network(
        data[:][0:-1],
        output_path = output_path,
        name = "SNN-0"
    )
for sample in result:
    harvest = sample["fire_rate"]
    if data[count][-1] == 0:
        temp0_mat += harvest
    else:
        temp1_mat += harvest

    save_heatmap(
        harvest,
        os.path.join(output_path, "label:" + str(data[count][-1]) + "fig " + str(count) + ".png"),
        plot_title="Spike rate"
    )
    plt.savefig(os.path.join(output_path, "label:" + str(data[count][-1]) + "fig " + str(count) + ".png"))
    count += 1

save_heatmap(
    temp0_mat,
    os.path.join(output_path, "overall_0.png"),
    plot_title="Spike rate (Overall for 0)"
)

save_heatmap(
    temp1_mat,
    os.path.join(output_path, "overall_1.png"),
    plot_title="Spike rate (Overall for 1)"
)

save_heatmap(
    temp0_mat + temp1_mat,
    os.path.join(output_path, "overall.png"),
    plot_title="Spike rate (Overall)"
)

# plt.imshow(harvest, cmap='hot', interpolation='nearest', aspect="auto")
# plt.show()

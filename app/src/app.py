import sys
import os

import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from setuptools.sandbox import save_argv
from tensorflow import keras
from brian2 import *
from random import random
from matplotlib.pyplot import *

from network import generate_network
from ann import function

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

    xlabel("Time partition index ({}ms each)".format(part_length))
    ylabel("Neuron index (i)")
    # xticks([
    #     i for i in range(1, int(run_time / part_length))
    # ])
    ax.set_title(plot_title)
    fig.tight_layout()
    plt.savefig(file_path)
    # plt.close(fig)


# Output directory control
# output_path = os.path.join(os.environ["OUTPUT_PATH"])

output_path = "../output/iris.7.out"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

run_time = 200
part_length = 10
G_N_number = 30 # G group number.

data_file = open("../data/iris-custom.data", "r")
data_lines = data_file.readlines()
data = list(map(lambda x: list(map(float, x.split(","))), data_lines))
data = np.array(data)

temp0_mat = np.zeros(shape=(30, 20)) # support more than two class
temp1_mat = np.zeros(shape=(30, 20))

result = generate_network(
    data[:, 0:-1],
    output_path = output_path,
    name = "SNN-0",
    tau_range = linspace(1, 1, G_N_number)*ms,
    run_time=run_time,
    part_length=part_length
)

# ANN will take place here
# Using output of SNN we will train an ANN
# And using the trained ANN we will do binary classification on test data

input_layer_N_num = (run_time / part_length) * G_N_number
hidden_layer_N_num = 30 #((run_time / part_length) - 1) or 2
output_layer_N_num = 2 # Generic output configuration will be implemented

function(
    result,
    data[:, -1],
    output_path,
    input_layer_N_num=input_layer_N_num,
    hidden_layer_N_num=hidden_layer_N_num,
    output_layer_N_num=output_layer_N_num
)

count = 0
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

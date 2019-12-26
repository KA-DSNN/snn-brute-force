import os
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

def function(
    snn_results,
    labels,
    output_path,
    input_layer_N_num = 600,
    hidden_layer_N_num = 19,
    output_layer_N_num = 2 # Generic output configuration will be implemented
):
    model = Sequential()
    model.add(
        Dense(units=input_layer_N_num, input_dim=input_layer_N_num, activation="relu")
    )
    model.add(
        Dense(units=hidden_layer_N_num, activation="relu")
    )
    model.add(
        Dense(units=output_layer_N_num, activation="relu")
    )
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    snn_results_x = list(map(lambda x: x["fire_rate"].ravel(), snn_results))
    snn_results_x = np.vstack(snn_results_x)

    print(snn_results[0]["fire_rate"])
    print(snn_results_x)

    X_train, X_test, y_train, y_test = train_test_split(
        snn_results_x,
        labels,
        test_size = 0.20,
        random_state = 42
    )

    model.fit(X_train, y_train, batch_size=1000, epochs=1000, verbose=2)
    model.save(
        os.path.join(output_path, "ann-model.h5")
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print("loss: %.2f%%; acc: %.2f%%" % (loss * 100, acc * 100))
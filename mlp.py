#! /bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

DATA_FILE_PATH = "./data/data.csv"


def draw_plot(title, x_label, y_label, data, output):
    plt.figure()
    plt.bar([str(key) for key in data.keys()], data.values())

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([0.9, 1.0])

    plt.savefig(output)


def get_key_with_highest_score(input_dict):
    best_key = ()
    best_score = 0
    for key, score in input_dict.items():
        if score > best_score:
            best_score = score
            best_key = key

    return best_key


def read_and_preprocess_data(path):
    data = pd.read_csv(path)
    data = data.drop("id", axis=1)

    data["diagnosis"] = [0 if value == "M" else 1 for value in data["diagnosis"]]

    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = scaler.fit_transform(data)

    diagnosis = np.array([entry[0] for entry in data])
    parameters = np.array([entry[1:-1] for entry in data])

    return (diagnosis, parameters)


def perform_mlp_classification_by_hiden_layer(diagnosis_data, parameters):
    hidden_layer_sizes = [(30,), (30, 30), (45, 70), (60,), (60, 60), (100,), (10, 40)]
    scores = {}

    for layer in hidden_layer_sizes:
        mlp = MLPClassifier(hidden_layer_sizes=layer, random_state=1410, max_iter=400)
        score = cross_val_score(
            mlp, parameters, diagnosis_data, scoring="balanced_accuracy"
        ).mean()
        scores[layer] = score

    return scores


def perform_mlp_classification_activation(
    diagnosis_data, parameters, hidden_layer_size
):
    activation_methods = ["relu", "identity", "logistic", "tanh"]
    scores = {}

    for method in activation_methods:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            random_state=1410,
            max_iter=400,
            activation=method,
        )
        score = cross_val_score(
            mlp, parameters, diagnosis_data, scoring="balanced_accuracy"
        ).mean()
        scores[method] = score

    return scores


def perform_mlp_classification_by_solver(
    diagnosis_data, parameters, hidden_layer_size, activation_method
):
    solvers = ["lbfgs", "sgd", "adam"]
    scores = {}

    for solver in solvers:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_size,
            random_state=1410,
            max_iter=400,
            activation=activation_method,
            solver=solver,
        )
        score = cross_val_score(
            mlp, parameters, diagnosis_data, scoring="balanced_accuracy"
        ).mean()
        scores[solver] = score

    return scores


def main():
    diagnosis, parameters = read_and_preprocess_data(DATA_FILE_PATH)

    hidden_layer_scores = perform_mlp_classification_by_hiden_layer(
        diagnosis_data=diagnosis, parameters=parameters
    )
    best_hidden_layer = get_key_with_highest_score(hidden_layer_scores)

    draw_plot(
        "MLPClassifier by hidden_layer_sizes",
        "hidden_layer_sizes",
        "scores",
        hidden_layer_scores,
        "output/mlpclassifier_by_hidden_layers",
    )

    activation_scores = perform_mlp_classification_activation(
        diagnosis_data=diagnosis,
        parameters=parameters,
        hidden_layer_size=best_hidden_layer,
    )
    best_activation_method = get_key_with_highest_score(activation_scores)

    draw_plot(
        "MLPClassifier by activation method",
        "activation method",
        "scores",
        activation_scores,
        "output/mlpclassifier_by_activation_method",
    )

    solver_scores = perform_mlp_classification_by_solver(
        diagnosis_data=diagnosis,
        parameters=parameters,
        hidden_layer_size=best_hidden_layer,
        activation_method=best_activation_method,
    )

    draw_plot(
        "MLPClassifier by solver",
        "solver",
        "scores",
        solver_scores,
        "output/mlpclassifier_by_solver",
    )


if __name__ == "__main__":
    main()

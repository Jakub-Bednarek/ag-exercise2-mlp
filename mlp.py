#! /bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

DATA_FILE_PATH = "./data/data.csv"

def get_best_layer_parameter(input_dict):
    best_layer = ()
    best_score = 0
    for layer, score in input_dict.items():
        if score > best_score:
            best_score = score
            best_layer = layer

    return best_layer

def read_and_preprocess_data(path):
    data = pd.read_csv(path)
    data = data.drop('id', axis=1)

    data['diagnosis'] = [ 0 if value == 'M' else 1 for value in data['diagnosis'] ]

    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = scaler.fit_transform(data)

    diagnosis = np.array([ entry[0] for entry in data ])
    parameters = np.array([ entry[1:-1] for entry in data ])

    return (diagnosis, parameters)

def perform_mlp_classification_by_hiden_layer(diagnosis_data, parameters):
    hidden_layer_sizes=[(30,), (30, 30), (45, 70), (60,), (60, 60), (100,), (10, 40)]
    scores = {}

    for layer in hidden_layer_sizes:
        mlp = MLPClassifier(hidden_layer_sizes=layer, random_state=1410, max_iter=400)
        score = cross_val_score(mlp, parameters, diagnosis_data, scoring='balanced_accuracy').mean()
        scores[layer] = score
        
    return scores

def perform_mlp_classification_activation(diagnosis_data, parameters, hidden_layer_size):
    activation_methods = ["relu", "identity", "logistic", "tanh"]
    scores = {}

    for method in activation_methods:
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, random_state=1410, max_iter=400, activation=method)
        score = cross_val_score(mlp, parameters, diagnosis_data, scoring='balanced_accuracy').mean()
        scores[method] = score
        
    return scores

def main():
    diagnosis, parameters = read_and_preprocess_data(DATA_FILE_PATH)

    hidden_layer_scores = perform_mlp_classification_by_hiden_layer(diagnosis_data=diagnosis, parameters=parameters)
    best_hidden_layer_classification_layer = get_best_layer_parameter(hidden_layer_scores)
    
    print(hidden_layer_scores)
    print(best_hidden_layer_classification_layer)

    activation_scores = perform_mlp_classification_activation(diagnosis_data=diagnosis, parameters=parameters, hidden_layer_size=best_hidden_layer_classification_layer)
    best_activation_classification_activation = get_best_layer_parameter(activation_scores)
    
    print(activation_scores)
    print(best_activation_classification_activation)

if __name__ == "__main__":
    main()
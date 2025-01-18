#! /bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

DATA_FILE_PATH = "./data/data.csv"

def read_and_preprocess_data(path):
    data = pd.read_csv(path)
    data = data.drop('id', axis=1)

    data['diagnosis'] = [ 0 if value == 'M' else 1 for value in data['diagnosis'] ]

    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = scaler.fit_transform(data)

    diagnosis = np.array([ entry[0] for entry in data ])
    parameters = np.array([ entry[1:-1] for entry in data ])

    return (diagnosis, parameters)

def perform_mlp_classification(diagnosis_data, parameters):
    hidden_layer_sizes = [(30,), (30, 30), (45, 70), (60,), (60, 60), (100,), (10, 40)]
    scores = {}

    for layer in hidden_layer_sizes:
        mlp = MLPClassifier(hidden_layer_sizes=layer, random_state=1410, max_iter=400)
        score = cross_val_score(mlp, parameters, diagnosis_data, scoring='balanced_accuracy').mean()
        scores[layer] = score
        
    return scores

def main():
    diagnosis, parameters = read_and_preprocess_data(DATA_FILE_PATH)

    perform_mlp_classification(diagnosis_data=diagnosis, parameters=parameters)

if __name__ == "__main__":
    main()
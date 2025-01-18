#! /bin/python3

import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_FILE_PATH = "./data/data.csv"

def read_and_preprocess_data(path):
    data = pd.read_csv(path)
    data = data.drop('id', axis=1)

    data['diagnosis'] = [ 0 if value == 'M' else 1 for value in data['diagnosis'] ]

    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = scaler.fit(data).transform(data)

    diagnosis = np.array([ entry[0] for entry in data ])
    parameters = np.matrix([ entry[1:-1] for entry in data ])

    return (diagnosis, parameters)

def main():
    diagnosis, parameters = read_and_preprocess_data(DATA_FILE_PATH)

    print(diagnosis)
    print(parameters)

if __name__ == "__main__":
    main()
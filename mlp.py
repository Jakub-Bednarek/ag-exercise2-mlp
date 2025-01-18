#! /bin/python3

import numpy as np
import pandas as pd

DATA_FILE_PATH = "./data/data.csv"

def read_and_preprocess_data(path):
    data = pd.read_csv(path)
    data = data.drop('id', axis=1)

    data['diagnosis'] = [ 0 if value == 'M' else 1 for value in data['diagnosis'] ]

    return data

def main():
    data = read_and_preprocess_data(DATA_FILE_PATH)
    print(data['diagnosis'])
    # print(data)

if __name__ == "__main__":
    main()
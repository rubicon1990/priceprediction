import numpy as np
import pandas as pd
import math
import os


# function for normalization of data
def normalize_data(pdf, normalizer):
    values = normalizer.fit_transform(pdf.values)
    return pd.DataFrame(values, columns=pdf.columns)


# function to create train, validation, test data given stock data and sequence length
def extract_data(stock, seq_length):
    """
    :param stock: stock in ascending order by date, pandas dataframe
    :param seq_length:
    :return:
    """
    raw_data = stock.values
    _data = []

    # create all possible sequences of length seq_length
    for index in range(raw_data.shape[0] - seq_length):
        _data.append(raw_data[index: index + seq_length])
    return np.array(_data)


def train_valid_test_split(data_set, valid_set_size_percentage, test_set_size_percentage):
    _len = data_set.shape[0]
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * _len))
    test_set_size = int(np.round(test_set_size_percentage / 100 * _len))
    train_set_size = _len - valid_set_size - test_set_size

    _x_train = data_set[:train_set_size, :-1, :]
    _y_train = data_set[:train_set_size, -1, :]

    _x_valid = data_set[train_set_size:train_set_size + valid_set_size, :-1, :]
    _y_valid = data_set[train_set_size:train_set_size + valid_set_size, -1, :]

    _x_test = data_set[train_set_size + valid_set_size:, :-1, :]
    _y_test = data_set[train_set_size + valid_set_size:, -1, :]

    return [_x_train, _y_train, _x_valid, _y_valid, _x_test, _y_test]
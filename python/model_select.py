import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU
import tensorflow.keras.metrics as metrics
from python.prepare_data import *
from python.utils import *
from python.nn_engine import *

"""
Most ticker longer than 1000 values

"""


def prepare_data(data_df, normalizer):
    raw_stock_price = data_df[data_df.symbol == ticker][['open', 'high', 'low', 'close', 'volume']]
    sorted_stock_price = raw_stock_price.sort_index()
    norm_stock_price = normalize_data(sorted_stock_price, normalizer)
    data = extract_data(norm_stock_price, seq_len)
    return train_valid_test_split(data, valid_size_percent, test_size_percent)


def train_model(model):

    _model = model(h_dim, dropout_rate, output_dim)
    _model.compile(optimizer=optimizer,
                   loss="mse",
                   metrics=["mae", "mape", "mse"])

    return _model, _model.fit(x_train, y_train,
                              batch_size=batch_size, epochs=n_epochs,
                              shuffle=True,
                              validation_split=validation_split,
                              verbose=2)


def rec_per_param(model):
    # Out-of-sample test (Evaluation)
    y_test_pred = model(x_test)

    true_price = sc_norm.inverse_transform(y_test)
    pred_price = sc_norm.inverse_transform(y_test_pred)

    true_open = true_price[:, 0]
    pred_open = pred_price[:, 0]

    # Add to output
    train_loss = history.history['loss'][-1]
    train_mae = history.history['mae'][-1]
    train_mape = history.history['mape'][-1]
    train_mse = history.history['mse'][-1]

    valid_loss = history.history['val_loss'][-1]
    valid_mae = history.history['val_mae'][-1]
    valid_mape = history.history['val_mape'][-1]
    valid_mse = history.history['val_mse'][-1]

    test_mae = metrics.mean_absolute_error(true_open, pred_open).numpy().mean()
    test_mape = metrics.mean_absolute_percentage_error(true_open, pred_open).numpy().mean()
    test_mse = metrics.mean_squared_error(true_open, pred_open).numpy().mean()

    return [h_dim, learning_rate,
            train_loss, train_mae, train_mape, train_mse,
            valid_loss, valid_mae, valid_mape, valid_mse,
            test_mae, test_mape, test_mse]


def save():
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    for nm in sorted(output.keys()):
        res_df = pd.DataFrame(output[nm], columns=columns)
        res_df.to_excel(writer, sheet_name=nm)

    writer.save()


if __name__ == "__main__":

    # parameters
    n_epochs = 100
    batch_size = 32
    dropout_rate = 0.2

    # split data in 80%/10%/10% train/validation/test sets
    valid_size_percent = 0
    test_size_percent = 10
    validation_split = 0.2

    # dir and output column
    input_path = '../input/prices-split-adjusted.csv'
    output_path = '../output/result_20201008.xlsx'

    columns = ['h_dim', 'learning_rate',
               'train_loss', 'train_mae', 'train_mape', 'train_mse',
               'valid_loss', 'valid_mae', 'valid_mape', 'valid_mse',
               'test_mae', 'test_mape', 'test_mse']

    # Load data
    df = pd.read_csv(input_path, index_col=0)
    param_grad = {
        'ticker': ['EQIX', 'JPM', 'R', 'HES', 'COST'],
        'seq_len': [21, 31, 45, 61],
        'h_dim': [10, 15, 20, 25, 30],
        'learning_rate': [0.001, 0.01, 0.1]
    }

    # param_grad = {
    #     'ticker': ['EQIX', 'JPM'],
    #     'seq_len': [45, 61],
    #     'h_dim': [10, 15],
    #     'learning_rate': [0.01]
    # }

    output = {}
    for p_dict in ParameterGrid(param_grad):
        ticker = p_dict['ticker']
        seq_len = p_dict['seq_len']
        h_dim = p_dict['h_dim']
        learning_rate = p_dict['learning_rate']

        sheet_nm = '{0}_{1}'.format(ticker, seq_len)
        if sheet_nm not in output:
            output[sheet_nm] = []

        # Prepare data
        sc_norm = MinMaxScaler(feature_range=(0, 1))
        x_train, y_train, x_valid, y_valid, x_test, y_test = \
            prepare_data(df, sc_norm)

        # Model training
        output_dim = y_train.shape[-1]
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        nn_model, history = train_model(gru)

        # Output per parameter
        output[sheet_nm].append(rec_per_param(nn_model))

    # Save output to sheet
    save()




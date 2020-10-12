import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import datetime
import os
import gc
import csv
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


# def save():
#     writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
#
#     for nm in sorted(output.keys()):
#         res_df = pd.DataFrame(output[nm], columns=columns)
#         res_df.to_excel(writer, sheet_name=nm)
#
#     writer.save()


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

    columns = ['ticker', 'seq_len', 'h_dim', 'learning_rate',
               'train_loss', 'train_mae', 'train_mape', 'train_mse',
               'valid_loss', 'valid_mae', 'valid_mape', 'valid_mse',
               'test_mae', 'test_mape', 'test_mse']

    param_grad = {
        'ticker': ['EQIX', 'JPM', 'R', 'HES', 'COST'],
        'seq_len': [16, 21, 31],
        'h_dim': [10, 15, 20, 25, 30, 50],
        'learning_rate': [0.001, 0.01, 0.1]
    }

    # param_grad = {
    #     'ticker': ['R', 'JPM'],
    #     'seq_len': [31],
    #     'h_dim': [15],
    #     'learning_rate': [0.1]
    # }

    # Load data
    df = pd.read_csv(input_path, index_col=0)
    sc_norm = MinMaxScaler(feature_range=(0, 1))

    # output = {}
    with open('../output/result_20201009.csv', 'a') as _f:
        csv_write = csv.writer(_f)
        csv_write.writerow(columns)

        for p_dict in ParameterGrid(param_grad):
            ticker = p_dict['ticker']
            seq_len = p_dict['seq_len']
            h_dim = p_dict['h_dim']
            learning_rate = p_dict['learning_rate']

            print('{0} - {1} - {2} - {3}'.format(ticker, seq_len, h_dim, learning_rate))

            # sheet_nm = '{0}_{1}'.format(ticker, seq_len)
            # if sheet_nm not in output:
            #     output[sheet_nm] = []

            # Prepare data
            x_train, y_train, x_valid, y_valid, x_test, y_test = \
                prepare_data(df, sc_norm)

            # Model training
            output_dim = y_train.shape[-1]
            optimizer = tf.keras.optimizers.Adam(learning_rate)

            tf.keras.backend.clear_session()
            # nn_model = \
            #     tf.keras.Sequential([GRU(h_dim, activation='tanh'),
            #                          Dropout(dropout_rate),
            #                          Dense(output_dim)])

            # Construct and compile
            inputs = tf.keras.Input(shape=(None, output_dim))
            x = tf.keras.layers.GRU(h_dim, activation=tf.nn.tanh)(inputs)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            outputs = tf.keras.layers.Dense(output_dim)(x)

            nn_model = tf.keras.Model(inputs, outputs)

            nn_model.compile(optimizer=optimizer,
                             loss="mse",
                             metrics=["mae", "mape", "mse"])

            history = nn_model.fit(x_train, y_train,
                                   batch_size=batch_size, epochs=n_epochs,
                                   shuffle=True,
                                   validation_split=validation_split,
                                   verbose=2)

            # Output per parameter
            # output[sheet_nm].append(rec_per_param(nn_model))
            csv_write.writerow([ticker, seq_len] + rec_per_param(nn_model))

            del nn_model
            gc.collect()

    # Save output to sheet
    # save()




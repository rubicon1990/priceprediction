import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
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


if __name__ == "__main__":

    # parameters
    ticker = 'EQIX'
    seq_len = 30
    h_dim = 10
    n_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    dropout_rate = 0.2

    # split data in 80%/10%/10% train/validation/test sets
    valid_size_percent = 0
    test_size_percent = 10
    validation_split = 0.2

    # dir for model save
    saved_model_path = "../saved_model"

    # Data preparation
    df = pd.read_csv("../input/prices-split-adjusted.csv", index_col=0)
    raw_stock_price = df[df.symbol == ticker][['open', 'high', 'low', 'close', 'volume']]

    sc = MinMaxScaler(feature_range=(0, 1))

    sorted_stock_price = raw_stock_price.sort_index()
    norm_stock_price = normalize_data(sorted_stock_price, sc)

    data = extract_data(norm_stock_price, seq_len)
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        train_valid_test_split(data, valid_size_percent, test_size_percent)

    # Model training
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    nn_model = gru(h_dim, dropout_rate, data.shape[-1])
    nn_model.compile(optimizer=optimizer,
                     loss="mse",
                     metrics=["mae", "mape", "mse"])

    history = nn_model.fit(x_train, y_train,
                           batch_size=batch_size, epochs=n_epochs,
                           shuffle=True,
                           validation_split=validation_split,
                           verbose=2)

    # Out-of-sample test (Evaluation)
    y_test_pred = nn_model(x_test)

    true_price = sc.inverse_transform(y_test)
    pred_price = sc.inverse_transform(y_test_pred)

    true_open = true_price[:, 0]
    pred_open = pred_price[:, 0]

    mse = metrics.mean_squared_error(true_open, pred_open).numpy().mean()
    rmse = math.sqrt(mse)
    mae = metrics.mean_absolute_error(true_open, pred_open).numpy().mean()
    mape = metrics.mean_absolute_percentage_error(true_open, pred_open).numpy().mean()
    print('MSE - %.6f' % mse)
    print('RMSE - %.6f' % rmse)
    print('MAE - %.6f' % mae)
    print('MAPE - %.6f' % mape)

    # Loss plotting
    loss_history = {'train': history.history['loss'],
                    'valid': history.history['val_loss']}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_loss_history(loss_history, n_epochs)

    plt.subplot(1, 2, 2)
    plt.plot(true_open, label='test true', color='black')
    plt.plot(pred_open, label='test prediction', color='green')
    plt.title('test open stock price')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Model save
    model_save(nn_model, saved_model_path)

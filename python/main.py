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

tf.keras.backend.set_floatx('float64')


if __name__ == "__main__":

    # parameters
    ticker = 'EQIX'
    seq_len = 20
    n_epochs = 1
    batch_size = 32
    learning_rate = 0.02

    # split data in 80%/10%/10% train/validation/test sets
    valid_size_percent = 10
    test_size_percent = 10

    # dir for model save
    saved_model_path = "../saved_model"

    # Data preparation
    df = pd.read_csv("../input/prices-split-adjusted.csv", index_col=0)
    raw_stock_price = df[df.symbol == ticker][['open', 'high', 'low', 'close']]

    sc = MinMaxScaler(feature_range=(0, 1))

    sorted_stock_price = raw_stock_price.sort_index()
    stock_price, sc = normalize_data(sorted_stock_price, sc)

    data = extract_data(stock_price, seq_len)
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        train_valid_test_split(data, valid_size_percent, test_size_percent)

    # Model training
    model = tf.keras.Sequential([
        SimpleRNN(2, activation='tanh'),
        Dropout(0.2),
        Dense(data.shape[-1])
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_object = tf.keras.losses.mean_squared_error

    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    loss_history = {'train': [], 'valid': []}

    iteration = 0

    mse_train = tf.math.reduce_mean(loss_object(y_train, model(x_train)))
    mse_valid = tf.math.reduce_mean(loss_object(y_valid, model(x_valid)))
    print('%d epochs, %d iterations: MSE train/valid = %.6f/%.6f'
          % (0, iteration, mse_train, mse_valid))

    for epoch in range(1, n_epochs + 1):
        # TODO: set buffer_size and seed
        ds = data_train.shuffle(x_train.shape[0]).batch(batch_size)

        for x, y in ds:
            if x.shape[0] == batch_size:
                with tf.GradientTape() as tape:
                    predictions = model(x)
                    loss = loss_object(y, predictions)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                loss_history['train'].append(loss_object(y_train, model(x_train)).numpy().mean())
                loss_history['valid'].append(loss_object(y_valid, model(x_valid)).numpy().mean())
                iteration += 1

        if epoch % 5 == 0:
            mse_train = tf.math.reduce_mean(loss_object(y_train, model(x_train)))
            mse_valid = tf.math.reduce_mean(loss_object(y_valid, model(x_valid)))
            print('%d epochs, %d iterations: MSE train/valid = %.6f/%.6f'
                  % (epoch, iteration, mse_train, mse_valid))

    model.summary()

    # Loss plotting
    plot_loss_history(loss_history, n_epochs)

    # Out-of-sample test (Evaluation)
    y_test_pred = model(x_test)
    mse = metrics.mean_squared_error(y_test, y_test_pred).numpy().mean()
    rmse = math.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_test_pred).numpy().mean()
    mape = metrics.mean_absolute_percentage_error(y_test, y_test_pred).numpy().mean()
    print('MSE - %.6f' % mse)
    print('RMSE - %.6f' % rmse)
    print('MAE - %.6f' % mae)
    print('MAPE - %.6f' % mape)

    plt.plot(y_test[:, 0], label='test true', color='black')
    plt.plot(y_test_pred[:, 0], label='test prediction', color='green')
    plt.title('test open stock price')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best')
    plt.show()

    # Model save
    model_save(model, x_train, saved_model_path)

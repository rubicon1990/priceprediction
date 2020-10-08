import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU


def simple_rnn(h_dim, dropout_rate, output_dim):
    return tf.keras.Sequential([
        SimpleRNN(h_dim, activation='tanh'),
        Dropout(dropout_rate),
        Dense(output_dim)])


def lstm(h_dim, dropout_rate, output_dim):
    return tf.keras.Sequential([
        LSTM(h_dim, activation='tanh'),
        Dropout(dropout_rate),
        Dense(output_dim)])


def gru(h_dim, dropout_rate, output_dim):
    return tf.keras.Sequential([
        GRU(h_dim, activation='tanh'),
        Dropout(dropout_rate),
        Dense(output_dim)])

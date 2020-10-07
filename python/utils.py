import tensorflow as tf
import numpy as np
import math
import sklearn
import datetime
import os
import matplotlib.pyplot as plt


def plot_loss_history(loss_history, n_epochs):
    plt.plot(loss_history['train'], label='training loss', color='black')
    plt.plot(loss_history['valid'], label='validation loss', color='green')
    plt.title('loss history - %d epochs' % n_epochs)
    plt.xlabel('iteration')
    plt.ylabel('mse')
    plt.legend(loc='best')
    plt.show()


def plot_timeseries(timeseries):
    pass


def model_save(model, input, path):
    model._set_inputs(input)
    model.save(path)


def model_load(path):
    return tf.keras.models.load_model(path)



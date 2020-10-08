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
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.legend(loc='best')


def model_save(model, path, overwrite=True):
    if os.path.exists(path):
        if overwrite:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        else:
            raise FileExistsError('Directory already exists: %s'
                                  % os.path.abspath(path))
    model.save(path)


def model_load(path):
    return tf.keras.models.load_model(path)



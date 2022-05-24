#!/usr/bin/env python3
# coding: utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, activations, losses, optimizers, backend
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import random as rn
import sys
import click
from sklearn.preprocessing import MinMaxScaler

MINMAX_SCALER = True
PATIENCE = 1

# set seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)

def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def approx_relu(z):
    return -0.0061728 * z**3 + 0.092593 * z**2 + 0.59259 * z + 0.49383

def lower_network():
    model = models.Sequential()
    model.add(
        layers.Conv1D(filters=1,
                      kernel_size=9,
                      strides=1,
                      padding="same",
                      input_shape=(768, 1),
                      use_bias=False))
    # lower layers
    model.add(layers.Flatten())
    model.add(layers.Dense(768, activation=approx_relu))
    model.add(layers.Reshape((768, 1)))
    model.add(layers.AveragePooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(766))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def full_network(lower_network):
    model = models.Sequential()

    # add and freeze lower_network
    for layer in lower_network.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    # upper layers
    model.add(layers.Dense(766, activation=activations.relu))
    model.add(layers.Reshape((766, 1)))
    model.add(layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def full_network_arch():
    model = models.Sequential()

    # lower layers
    model.add(
        layers.Conv1D(filters=1,
                      kernel_size=9,
                      strides=1,
                      padding="same",
                      input_shape=(768, 1),
                      use_bias=False))
    # lower layers
    model.add(layers.Flatten())
    model.add(layers.Dense(768, activation=approx_relu))
    model.add(layers.Reshape((768, 1)))
    model.add(layers.AveragePooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(766))

    # upper layers
    model.add(layers.Dense(766, activation=activations.relu))
    model.add(layers.Reshape((766, 1)))
    model.add(layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


def train_lower_network(x_train, y_train, x_test, y_test, epochs):
    model = lower_network()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)

    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        verbose=0)

    return model


def train_network(x_train, y_train, x_test, y_test, epochs, lower):
    model = full_network(lower)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)

    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        verbose=0)

    return model


def saveModel(model, filename):
    model.save_weights(filename, save_format="h5")


def loadModel(filename):
    model = full_network_arch()
    model.load_weights(filename)
    return model


def preprocessDataMinMax(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset.iloc[:, :-1])
    dataset.iloc[:, :-1] = scaler.transform(dataset.iloc[:, :-1])
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    return X, Y


def preprocessData(dataset):
    if MINMAX_SCALER == True:
        return preprocessDataMinMax(dataset)
    inputs, labels = dataset.iloc[:, :-1].values, dataset.iloc[:, -1]
    return inputs, labels


def train(train_dataset,
          epochs,
          splits,
          lower=False,
          lower_model=None):

    #Select data features and labels
    x, y = preprocessData(train_dataset)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=splits)

    max_acc = 0.
    average_acc = 0.
    best_model = None
    i = 0
    for train, test in kfold.split(x, y):
        i = i + 1
        print(
            "================================================================="
        )
        print("Split", i)
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        if lower:
            model = train_lower_network(x_train, y_train, x_test, y_test, epochs)
        else:
            model = train_network(x_train, y_train, x_test, y_test, epochs,
                              lower_model)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test F1 score:', score[2])
        print('Test precision:', score[3])
        print('Test recall: ', score[4])

        average_acc = average_acc + score[1]
        if score[1] > max_acc:
            best_model = model
            max_acc = score[1]

    print("=================================================================")
    average_acc = average_acc / splits
    print('Average accuracy:', average_acc)
    print('Best accuracy:', max_acc)

    return best_model

@click.command()
@click.option('--epochs', '-e', type=int, default=300)
@click.option('--splits', '-s', type=int, default=10)
@click.option('--source_dataset', '-d', type=str,
              default="")  # source dataset
@click.option('--target_dataset_train', '-t', type=str,
              default="")  # target dataset
@click.option('--target_dataset_test', '-l', type=str,
              default="")  # target dataset
@click.option('--model', '-m', type=str, default="")  # weights
@click.option('--classify', '-c', is_flag=True)
def main(epochs, splits, source_dataset, target_dataset_train,
         target_dataset_test, model, classify):

    if not os.path.exists(target_dataset_test):
        print('[x] Error: target dataset test file does not exist.')
        sys.exit(-1)

    #Read the data here
    targetdata_test = pd.read_csv(target_dataset_test)

    if classify:
        # Full model for prediction
        if not os.path.exists(model):
            print('[x] Error: model does not exist.')
            sys.exit(-1)
        mod = loadModel(model)
        mod.summary()
    else:
        if not os.path.exists(source_dataset):
            print('[x] Error: source dataset train file does not exist.')
            sys.exit(-1)

        if not os.path.exists(target_dataset_train):
            print('[x] Error: target dataset train file does not exist.')
            sys.exit(-1)

        sourcedata = pd.read_csv(source_dataset)
        targetdata_train = pd.read_csv(target_dataset_train)

        # train lower
        print(
            "================================================================="
        )
        print("Training lower network:")
        mod = lower_network()
        mod.summary()
        mod = train(sourcedata, epochs, splits, True)

        print()
        print(
            "================================================================="
        )
        print("Finetuning upper layer:")
        # finetune upper
        tmp = full_network(mod)
        tmp.summary()
        mod = train(targetdata_train, epochs, splits, False, mod)
        saveModel(mod, model)
        print("")

    print("Predicting whole test dataset using best model:")
    inputs, labels = preprocessData(targetdata_test)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
    score = mod.evaluate(inputs, labels, verbose=0)

    print('Loss:', score[0])
    print('Accuracy:', score[1])
    print('F1 score:', score[2])
    print('Precision:', score[3])
    print('Recall: ', score[4])


if __name__ == '__main__':
    sys.exit(main())

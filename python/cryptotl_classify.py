#!/usr/bin/env python3
# coding: utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, activations, losses, optimizers, backend
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


def upper_network():
    model = models.Sequential()

    # upper layers
    model.add(
        layers.Dense(766, activation=activations.relu, input_shape=(766, )))
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

def splitModel(model):
    lower = models.Sequential()

    for layer in model.layers[:8]:
        lower.add(layer)

    lower.compile(optimizer=optimizers.Adam(),
                loss=losses.binary_crossentropy,
                metrics=['accuracy', f1_m, precision_m, recall_m])

    upper = models.Sequential()

    for layer in model.layers[8:]:
        upper.add(layer)

    upper.compile(optimizer=optimizers.Adam(),
                loss=losses.binary_crossentropy,
                metrics=['accuracy', f1_m, precision_m, recall_m])

    return lower, upper

def combineModels(lower, upper):
    model = models.Sequential()

    # add and freeze lower_network
    for layer in lower.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    # add upper layers
    for layer in upper.layers:
        model.add(layer)

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


def loadLowerModel(filename):
    model = lower_network()
    model.load_weights(filename)
    return model


def loadUpperModel(filename):
    model = upper_network()
    model.load_weights(filename)
    return model

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


def storeLowerHeWeights(model, filename):
    filter = model.layers[0].weights[0].numpy().flatten()
    mat1 = model.layers[2].weights[0].numpy()
    bias1 = model.layers[2].weights[1].numpy().flatten()
    mat2 = model.layers[7].weights[0].numpy()
    bias2 = model.layers[7].weights[1].numpy().flatten()
    with open(filename, "w") as outfile:
        writeConv1d(filter, outfile)
        writeDense(mat1, bias1, outfile)
        writeDense(mat2, bias2, outfile)


def storeUpperHeWeights(model, filename):
    mat1 = model.layers[0].weights[0].numpy()
    bias1 = model.layers[0].weights[1].numpy().flatten()
    mat2 = model.layers[5].weights[0].numpy()
    bias2 = model.layers[5].weights[1].numpy().flatten()
    with open(filename, "w") as outfile:
        writeDense(mat1, bias1, outfile)
        writeDense(mat2, bias2, outfile)


def writeConv1d(filter, outfile):
    # outfile must be open for writing
    outfile.write("Conv1d layer with " + str(1) + " filters of size " +
                  str(len(filter)) + ":\n")
    for i in range(1):
        outfile.write("Filter " + str(i) + "\n")
        for j in range(len(filter)):
            outfile.write(str(filter[j]) + "\n")


def writeDense(mat, bias, outfile):
    # outfile must be open for writing
    shape = mat.shape
    outfile.write("Dense layer of dimension " + str(shape) + ":\n")
    for j in range(shape[1]):
        outfile.write("Columns " + str(j) + "\n")
        for i in range(shape[0]):
            outfile.write(str(mat[i][j]) + "\n")

    outfile.write("Bias\n")
    for i in range(len(bias)):
        outfile.write(str(bias[i]) + "\n")


def combine_dense(weights1, biases1, weights2, biases2):
    weights = np.matmul(weights1, weights2)
    bias = np.dot(biases1, weights2) + biases2
    return weights, bias


def storeInputs(X, Y, filename):
    with open(filename, "w") as f:
        f.write(
            str(len(X)) + " datasets with " + str(len(X[0])) +
            " features each:\n")
        for i in range(len(X)):
            f.write("Dataset " + str(i) + "\n")
            for j in range(len(X[i])):
                f.write(str(X[i][j]) + "\n")
            f.write("Label " + str(i) + "\n")
            f.write(str(Y[i]) + "\n")


@click.command()
@click.option('--dataset', '-d', type=str, default="")  # source dataset
@click.option('--model', '-m', type=str, default="")  # weights
@click.option('--lower_model', '-l', type=str, default="")  # weights
@click.option('--upper_model', '-u', type=str, default="")  # weights
@click.option('--inputfile', '-i', type=str, default="")  # inputs for he
@click.option('--client_weightfile', '-c', type=str,
              default="")  # weights for he
@click.option('--server_weightfile', '-s', type=str,
              default="")  # weights for he
def main(dataset, model, lower_model, upper_model, inputfile, client_weightfile,
         server_weightfile):

    if not os.path.exists(dataset):
        print('[x] Error: dataset file does not exist.')
        sys.exit(-1)

    #Read the data here
    data = pd.read_csv(dataset)

    # models for prediction

    if model != "":
        if not os.path.exists(model):
            print('[x] Error: model does not exist.')
            sys.exit(-1)

        mod = loadModel(model)
        l_mod, u_mod = splitModel(mod)
    else:
        if not os.path.exists(lower_model):
            print('[x] Error: lower model does not exist.')
            sys.exit(-1)

        if not os.path.exists(upper_model):
            print('[x] Error: upper model does not exist.')
            sys.exit(-1)

        l_mod = loadLowerModel(lower_model)
        u_mod = loadUpperModel(upper_model)
        mod = combineModels(l_mod, u_mod)
    mod.summary()

    inputs, labels = preprocessData(data)

    if inputfile != "":
        storeInputs(inputs, labels, inputfile)

    if client_weightfile != "":
        storeUpperHeWeights(u_mod, client_weightfile)
    if server_weightfile != "":
        storeLowerHeWeights(l_mod, server_weightfile)

    print("Predicting whole dataset using whole model:")
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
    score = mod.evaluate(inputs, labels, verbose=0)

    print('Loss:', score[0])
    print('Accuracy:', score[1])
    print('F1 score:', score[2])
    print('Precision:', score[3])
    print('Recall: ', score[4])


if __name__ == '__main__':
    sys.exit(main())

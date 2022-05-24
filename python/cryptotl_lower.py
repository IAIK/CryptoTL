#!/usr/bin/env python3
# coding: utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, activations, losses, optimizers, backend
from enum import Enum
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


def saveModel(model, filename):
    model.save_weights(filename, save_format="h5")


def loadModel(filename):
    model = lower_network()
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


def train(train_dataset, epochs, splits): #Select data features and labels
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

        model = train_lower_network(x_train, y_train, x_test, y_test, epochs)

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


def storeHeWeights(model, filename, lastlayer=False):
    filter = model.layers[0].weights[0].numpy().flatten()
    mat1 = model.layers[2].weights[0].numpy()
    bias1 = model.layers[2].weights[1].numpy().flatten()
    mat2 = model.layers[7].weights[0].numpy()
    bias2 = model.layers[7].weights[1].numpy().flatten()
    if lastlayer == True:
        mat3 = model.layers[8].weights[0].numpy()
        bias3 = model.layers[8].weights[1].numpy().flatten()
        mat2, bias2 = combine_dense(mat2, bias2, mat3, bias3)
    with open(filename, "w") as outfile:
        writeConv1d(filter, outfile)
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
@click.option('--epochs', '-e', type=int, default=300)
@click.option('--splits', '-s', type=int, default=10)
@click.option('--train_dataset', '-t', type=str, default="")  # dataset
@click.option('--test_dataset', '-d', type=str, default="")  # dataset
@click.option('--model', '-m', type=str, default="")  # weights
@click.option('--classify', '-c', is_flag=True)
@click.option('--inputfile_train', '-i', type=str, default="")  # inputs for he
@click.option('--inputfile_test', '-j', type=str, default="")  # inputs for he
@click.option('--weightfile', '-w', type=str, default="")  # weights for he
@click.option('--weightclassificationfile', '-f', type=str,
              default="")  # weights
def main(epochs, splits, train_dataset, test_dataset, model, classify,
         inputfile_train, inputfile_test, weightfile,
         weightclassificationfile):
    if not os.path.exists(train_dataset):
        print('[x] Error: train dataset file does not exist.')
        sys.exit(-1)

    if not os.path.exists(test_dataset):
        print('[x] Error: test dataset file does not exist.')
        sys.exit(-1)

    # Read the data here
    train_data = pd.read_csv(train_dataset)
    test_data = pd.read_csv(test_dataset)

    if classify:
        # Full model for prediction
        if not os.path.exists(model):
            print('[x] Error: model does not exist.')
            sys.exit(-1)
        mod = loadModel(model)
        mod.summary()
    else:
        # train lower
        print(
            "================================================================="
        )
        print("Training lower network:")
        mod = lower_network()
        mod.summary()
        mod = train(train_data, epochs, splits)
        saveModel(mod, model)

    inputs, labels = preprocessData(train_data)

    if inputfile_train != "":
        storeInputs(inputs, labels, inputfile_train)

    inputs, labels = preprocessData(test_data)

    if inputfile_test != "":
        storeInputs(inputs, labels, inputfile_test)

    if weightfile != "":
        storeHeWeights(mod, weightfile)
    if weightclassificationfile != "":
        storeHeWeights(mod, weightclassificationfile, True)

    print("Predicting whole test dataset using best model:")
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
    score = mod.evaluate(inputs, labels, verbose=0)

    print('Loss:', score[0])
    print('Accuracy:', score[1])
    print('F1 score:', score[2])
    print('Precision:', score[3])
    print('Recall: ', score[4])


if __name__ == '__main__':
    sys.exit(main())

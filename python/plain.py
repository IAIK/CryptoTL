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


# just for debugging
# def predict(model, input):
    # if (len(input.shape) != 3):
    #     input = input.reshape(1, input.shape[0], 1)

    # # filter = model.layers[0].weights[0].numpy().flatten()
    # res = model.layers[0](input)  # conv1d
    # res = model.layers[1](res)  # flatten
    # # mat = model.layers[2].weights[0].numpy()
    # # bias = model.layers[2].weights[1].numpy().flatten()
    # # res = np.dot(res.numpy().flatten(), mat) + bias
    # res = model.layers[2](res)  # dense relu
    # res = model.layers[3](res)  # reshape
    # res = model.layers[4](res)  # max pool
    # res = model.layers[5](res)  # flatten
    # res = model.layers[6](res)  # dense
    # res = model.layers[7](res)  # dense
    # res = model.layers[8](res)  # reshape
    # res = model.layers[9](res)  # max pool
    # res = model.layers[10](res)  # flatten
    # res = model.layers[11](res)  # dense/sigmoid

    # res2 = model.predict(input)
    # assert res.numpy().flatten()[0] == res2.flatten()[0]


def network():
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
    model.add(layers.Dense(768, activation=activations.relu))
    model.add(layers.Reshape((768, 1)))
    model.add(layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(766))
    # model.add(layers.Dense(1, activation=activations.sigmoid))
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


def train_network(x_train, y_train, x_test, y_test, epochs):
    model = network()

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
    model = network()
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


def train(train_dataset, epochs, splits):
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

        model = train_network(x_train, y_train, x_test, y_test, epochs)

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
@click.option('--train_dataset', '-t', type=str, default="")  # dataset
@click.option('--test_dataset', '-d', type=str, default="")  # dataset
@click.option('--model', '-m', type=str, default="")  # weights
@click.option('--classify', '-c', is_flag=True)
def main(epochs, splits, train_dataset, test_dataset, model, classify):
    if not os.path.exists(train_dataset):
        print('[x] Error: train dataset file does not exist.')
        sys.exit(-1)

    if not os.path.exists(test_dataset):
        print('[x] Error: test dataset file does not exist.')
        sys.exit(-1)

    #Read the data here
    train_data = pd.read_csv(train_dataset)
    test_data = pd.read_csv(test_dataset)

    if classify:
        # model for prediction
        if not os.path.exists(model):
            print('[x] Error: model does not exist.')
            sys.exit(-1)
        mod = loadModel(model)
        mod.summary()
    else:
        # model from training
        mod = network()
        mod.summary()
        mod = train(train_data, epochs, splits)
        saveModel(mod, model)
        print("")

    print("Predicting whole test dataset using best model:")
    inputs, labels = preprocessData(test_data)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
    score = mod.evaluate(inputs, labels, verbose=0)

    print('Loss:', score[0])
    print('Accuracy:', score[1])
    print('F1 score:', score[2])
    print('Precision:', score[3])
    print('Recall: ', score[4])


if __name__ == '__main__':
    sys.exit(main())

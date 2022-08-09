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

def saveModel(model, filename):
    model.save_weights(filename, save_format="h5")


def loadModel(filename):
    model = upper_network()
    model.load_weights(filename)
    return model


def preprocessData(dataset):
    inputs, labels = dataset.iloc[:, :-1].values, dataset.iloc[:, -1]
    return inputs, labels


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


def train_upper_network(x_train, y_train, x_test, y_test, epochs):
    model = upper_network()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)

    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        verbose=0)

    return model


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

        model = train_upper_network(x_train, y_train, x_test, y_test, epochs)

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

def storeHeWeights(model, filename):
    mat1 = model.layers[0].weights[0].numpy()
    bias1 = model.layers[0].weights[1].numpy().flatten()
    mat2 = model.layers[5].weights[0].numpy()
    bias2 = model.layers[5].weights[1].numpy().flatten()
    with open(filename, "w") as outfile:
        writeDense(mat1, bias1, outfile)
        writeDense(mat2, bias2, outfile)


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


@click.command()
@click.option('--epochs', '-e', type=int, default=300)
@click.option('--splits', '-s', type=int, default=10)
@click.option('--train_features', '-t', type=str, default="")  # HE output
@click.option('--test_features', '-d', type=str, default="")  # HE output
@click.option('--model', '-m', type=str, default="")  # weights
@click.option('--classify', '-c', is_flag=True)
@click.option('--weightfile', '-w', type=str, default="")  # weights for he
def main(epochs, splits, train_features, test_features, model, classify, weightfile):
    if not os.path.exists(train_features):
        print('[x] Error: train features file does not exist.')
        sys.exit(-1)

    if not os.path.exists(test_features):
        print('[x] Error: test features file does not exist.')
        sys.exit(-1)

    #Read the data here
    train_data = pd.read_csv(train_features)
    test_data = pd.read_csv(test_features)

    if classify:
        # Full model for prediction
        if not os.path.exists(model):
            print('[x] Error: model does not exist.')
            sys.exit(-1)
        mod = loadModel(model)
        mod.summary()
    else:
        # finetune upper
        print(
            "================================================================="
        )
        print("Finetuning upper layer:")
        # finetune upper
        mod = upper_network()
        mod.summary()
        mod = train(train_data, epochs, splits)
        saveModel(mod, model)

    if weightfile != "":
        storeHeWeights(mod, weightfile)

    print("Predicting whole test dataset using best model:")
    inputs, labels = preprocessData(test_data)
    score = mod.evaluate(inputs, labels, verbose=0)

    print('Loss:', score[0])
    print('Accuracy:', score[1])
    print('F1 score:', score[2])
    print('Precision:', score[3])
    print('Recall: ', score[4])


if __name__ == '__main__':
    sys.exit(main())

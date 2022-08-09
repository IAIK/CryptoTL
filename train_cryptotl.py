#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import Popen, PIPE, STDOUT
import sys
import os
import click
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

PATH = "cpp"
BUILD_PATH = "cpp/build"
BINARY = "cpp/bin/main"
PYTHON = "python3"

MINMAX_SCALER = True

SERVER_WEIGHTS_CPP = "server_weights.in"
INPUTS_CPP_TRAIN = "inputs_train.in"
INPUTS_CPP_TEST = "inputs_test.in"
HE_OUT_CPP = "he_out.csv"
HE_OUT_CPP_TRAIN = "he_out_train.csv"
HE_OUT_CPP_TEST = "he_out_test.csv"

LOWER_SCRIPT = "python/cryptotl_lower.py"
FINETUNE_SCRIPT = "python/cryptotl_finetune.py"

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


def cmake(accurate_he):
    print("CMake...")
    args = ["cmake"]
    args.append("-DSTORE_INPUTS=On")
    args.append("-DSERVER_ONLY=Off")
    if accurate_he:
        args.append("-DACCURATE_PARAMS=On")
    else:
        args.append("-DACCURATE_PARAMS=Off")
    args.append("-B{}".format(BUILD_PATH))
    args.append("-S{}".format(PATH))
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
            print(output)
            exit(-5)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


def build():
    print("Building cpp...")
    args = ["make", "-C", BUILD_PATH]
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
            print(output)
            exit(-5)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


def run_cpp_train(accurate_he):
    cmake(accurate_he)
    build()
    print("Running cpp...")
    args = [BINARY, SERVER_WEIGHTS_CPP, INPUTS_CPP_TRAIN]
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
        os.rename(HE_OUT_CPP, HE_OUT_CPP_TRAIN)
        print(output)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


def run_cpp_test():
    print("Running cpp...")
    args = [BINARY, SERVER_WEIGHTS_CPP, INPUTS_CPP_TEST]
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
        os.rename(HE_OUT_CPP, HE_OUT_CPP_TEST)
        print(output)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


def python_lower(epochs, splits, source_dataset_train, source_dataset_test,
                 lower_model):
    print("Training lower layers...")
    args = [
        PYTHON, LOWER_SCRIPT, "-e",
        str(epochs), "-s", splits, "-t", source_dataset_train, "-d",
        source_dataset_test, "-m", lower_model, "-w", SERVER_WEIGHTS_CPP
    ]

    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
        print(output)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


def python_finetune(epochs, splits, upper_model):
    print("Finetuning upper layers..")
    args = [
        PYTHON, FINETUNE_SCRIPT, "-e",
        str(epochs), "-t", HE_OUT_CPP_TRAIN, "-d", HE_OUT_CPP_TEST, "-m",
        upper_model, "-s", splits
    ]
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
        print(output)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)


@click.command()
@click.option('--epochs', '-e', type=int, default=300)
@click.option('--splits', '-s', type=str, default=10)
@click.option('--source_dataset_train', '-d', type=str,
              default="")  # source dataset
@click.option('--target_dataset_train', '-t', type=str,
              default="")  # target dataset
@click.option('--source_dataset_test', '-k', type=str,
              default="")  # source dataset
@click.option('--target_dataset_test', '-m', type=str,
              default="")  # target dataset
@click.option('--lower_model', '-l', type=str,
              default="lower_model")  # weights
@click.option('--upper_model', '-u', type=str,
              default="upper_model")  # weights
@click.option('--accurate_he', '-a', is_flag=True)  # accurate HE params
def main(epochs, splits, source_dataset_train, target_dataset_train,
         source_dataset_test, target_dataset_test, lower_model, upper_model,
         accurate_he):
    if not os.path.exists(source_dataset_train):
        print('[x] Error: source dataset train file does not exist.')
        sys.exit(-1)
    if not os.path.exists(target_dataset_train):
        print('[x] Error: target dataset train file does not exist.')
        sys.exit(-1)
    if not os.path.exists(source_dataset_test):
        print('[x] Error: source dataset test file does not exist.')
        sys.exit(-1)
    if not os.path.exists(target_dataset_test):
        print('[x] Error: target dataset test file does not exist.')
        sys.exit(-1)

    target_data_train = pd.read_csv(target_dataset_train)
    inputs, labels = preprocessData(target_data_train)
    storeInputs(inputs, labels, INPUTS_CPP_TRAIN)

    target_data_test = pd.read_csv(target_dataset_test)
    inputs, labels = preprocessData(target_data_test)
    storeInputs(inputs, labels, INPUTS_CPP_TEST)

    python_lower(epochs, splits, source_dataset_train, source_dataset_test,
                 lower_model)
    run_cpp_train(accurate_he)
    run_cpp_test()
    python_finetune(epochs, splits, upper_model)


if __name__ == "__main__":
    sys.exit(main())

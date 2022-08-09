#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import Popen, PIPE, STDOUT
import sys
import os
import click

PATH = "cpp"
BUILD_PATH = "cpp/build"
BINARY = "cpp/bin/main"
PYTHON = "python3"

SERVER_WEIGHTS_CPP = "server_weights.in"
CLIENT_WEIGHTS_CPP = "client_weights.in"
INPUTS_CPP = "inputs.in"

SCRIPT = "python/cryptotl_classify.py"

def cmake(accurate_he):
    print("CMake...")
    args = ["cmake"]
    args.append("-DSTORE_INPUTS=Off")
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


def run_cpp(accurate_he):
    cmake(accurate_he)
    build()
    print("Running cpp...")
    args = [BINARY, SERVER_WEIGHTS_CPP, CLIENT_WEIGHTS_CPP, INPUTS_CPP]
    try:
        process = Popen(args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        output = process.communicate()[0].decode("utf-8")
        if process.returncode != 0:
            print("Exit code was {}".format(process.returncode))
        print(output)
    except Exception as ex:
        print("Exception: {}".format(ex))
        exit(-5)

def python_classify_model(dataset, model):
    print("Classify Python...")
    args = [
        PYTHON, SCRIPT, "-d", dataset, "-m", model,
        "-i", INPUTS_CPP, "-s", SERVER_WEIGHTS_CPP, "-c",
        CLIENT_WEIGHTS_CPP
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

def python_classify(dataset, lower_model, upper_model):
    print("Classify Python...")
    args = [
        PYTHON, SCRIPT, "-d", dataset, "-l", lower_model, "-u",
        upper_model, "-i", INPUTS_CPP, "-s", SERVER_WEIGHTS_CPP, "-c",
        CLIENT_WEIGHTS_CPP
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
@click.option('--dataset', '-d', type=str, default="")  # source dataset
@click.option('--model', '-m', type=str,
              default="")  # weights
@click.option('--lower_model', '-l', type=str,
              default="lower_model")  # weights
@click.option('--upper_model', '-u', type=str,
              default="upper_model")  # weights
@click.option('--accurate_he', '-a', is_flag=True)  # accurate HE params
def main(dataset, model, lower_model, upper_model, accurate_he):
    if not os.path.exists(dataset):
        print('[x] Error: dataset file does not exist.')
        sys.exit(-1)

    if model != "":
        python_classify_model(dataset, model)
    else:
        python_classify(dataset, lower_model, upper_model)
    run_cpp(accurate_he)


if __name__ == "__main__":
    sys.exit(main())

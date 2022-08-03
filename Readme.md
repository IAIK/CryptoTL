# CryptoTL

This repository contains the source code of the paper CryptoTL: Private, efficient and secure transfer learning [1].

[1] [https://arxiv.org/pdf/2205.11935.pdf](https://arxiv.org/pdf/2205.11935.pdf)

## Python Requirements

The required python3 libraries can be found in `requirements.txt`. Use the following command to install them all.

```bash
pip3 install -r requirements.txt
```

## SEAL -- Homomorphic Encryption Library

[SEAL](https://github.com/Microsoft/SEAL/) (version 4.0.0) is included as submodule in this repository. Use the following commands (from the root directory) to pull and build SEAL.

```bash
git submodule update --init --recursive
cd cpp/SEAL
mkdir build
cd build
cmake .. # replace with the following to use Clang: CC=clang CXX=clang++ cmake ..
make -j 6
cd ../../../
```

If you want to enable Intel HEXL (speedup if processors have the Intel AVX512-IFMA52 instruction set) replace the `cmake` command with the following command:

```bash
cmake -DSEAL_USE_INTEL_HEXL=On .. # replace with the following to use Clang: CC=clang CXX=clang++ cmake -DSEAL_USE_INTEL_HEXL=On ..
```

## Datasets

The (preprocessed) datasets we used in our experiments are attached to Github releases (e.g., [here](https://github.com/IAIK/CryptoTL/releases/tag/v1.0.0)), please download and extract them into the `datasets` folder.

| Dataset                                     |
|---------------------------------------------|
| `IMDB-train-SBERT.csv`                      |
| `IMDB-test-SBERT.csv`                       |
| `SBERT-youtube-combined.csv`                |
| `Twitter-train-SBERT.csv`                   |
| `Twitter-test-SBERT.csv`                    |

## Trained Weights

The folder `weights` contains models which have been trained to produce Table 3 in the paper.

| Weights                           |
|-----------------------------------|
| `cryptotl-imdb-to-imdb.bin`       |
| `cryptotl-imdb-to-twitter.bin`    |
| `cryptotl-imdb-to-yelp.bin`       |
| `cryptotl-imdb-to-youtube.bin`    |
| `cryptotl-twitter-to-imdb.bin`    |
| `cryptotl-twitter-to-twitter.bin` |
| `cryptotl-twitter-to-yelp.bin`    |
| `cryptotl-twitter-to-youtube.bin` |
| `cryptotl-youtube-to-imdb.bin`    |
| `cryptotl-youtube-to-twitter.bin` |
| `cryptotl-youtube-to-yelp.bin`    |
| `cryptotl-youtube-to-youtube.bin` |
|-----------------------------------|
| `cnn-full-imdb.bin`               |
| `cnn-full-twitter.bin`            |
| `cnn-full-youtube.bin`            |
|-----------------------------------|
| `cnn-upper-imdb.bin`              |
| `cnn-upper-twitter.bin`           |
| `cnn-upper-youtube.bin`           |

## Training the networks

`train_cryptotl.py`:

| Option               | Short | Value   | Information                                           | Default       |
|----------------------|-------|---------|-------------------------------------------------------|---------------|
| Epochs               | -e    | integer | Number of epochs to train                             | 300           |
| Splits               | -s    | integer | Splits for the K-fold Cross Validator                 | 10            |
| Source Dataset Train | -d    | Path    | Input path of the used source training dataset        | ""            |
| Target Dataset Train | -t    | Path    | Input path of the used target training dataset        | ""            |
| Source Dataset Test  | -k    | Path    | Input path of the used source test dataset            | ""            |
| Target Dataset Test  | -m    | Path    | Input path of the used target test dataset            | ""            |
| Lower Model          | -l    | Path    | Output path for the lower model                       | "lower_model" |
| Upper Model          | -u    | Path    | Output path for the upper model                       | "upper_model" |
| Accurate HE          | -a    | Flag    | Optional: If present, use more accurate HE parameters | false         |

Example:

```bash
python3 ./train_cryptotl.py -e 300 -s 10 -d ./datasets/IMDB-train-SBERT.csv -t ./datasets/Twitter-train-SBERT.csv -k ./datasets/IMDB-test-SBERT.csv -m ./datasets/Twitter-test-SBERT.csv
```

## Classification using CryptoTL

`classify_cryptotl.py`

| Option      | Short | Value  | Information                                           | Default       |
|-------------|-------|--------|-------------------------------------------------------|---------------|
| Dataset     | -d    | Path   | Input path of the dataset to classify                 | ""            |
| Model       | -m    | Path   | Input path for the full model                         | ""            |
| Lower Model | -l    | Path   | Input path for the lower model, if -m not specified   | "lower_model" |
| Upper Model | -u    | Path   | Input path for the upper model, if -m not specified   | "upper_model" |
| Accurate HE | -a    | Flag   | Optional: If present, use more accurate HE parameters | false         |

Example:

```bash
python3 ./classify_cryptotl.py -d ./datasets/Twitter-test-SBERT.csv
```

## Transfer Learning in Plain

`python/tl_plain.py`

| Option               | Short | Value   | Information                                          | Default       |
|----------------------|-------|---------|------------------------------------------------------|---------------|
| Epochs               | -e    | integer | Number of epochs to train                            | 300           |
| Splits               | -s    | integer | Splits for the K-fold Cross Validator                | 10            |
| Source Dataset       | -d    | Path    | Input path of the used source training dataset       | ""            |
| Target Dataset Train | -t    | Path    | Input path of the used target training dataset       | ""            |
| Target Dataset Test  | -l    | Path    | Input path of the used target test dataset           | ""            |
| Model                | -m    | Path    | Input/Output path for the full model                 | ""            |
| Classify             | -c    | Flag    | Optional: If present, load model instead of training | false         |

Example:

```bash
python3 python/tl_plain.py -e 300 -s 10 -d ./datasets/IMDB-train-SBERT.csv -t ./datasets/Twitter-train-SBERT.csv -l ./datasets/Twitter-test-SBERT.csv # training
python3 python/tl_plain.py -l ./datasets/Twitter-test-SBERT.csv -m model -c # classify
```

## Docker

To simplify building (without Intel HEXL) we have prepared a docker file, which installs all dependencies, builds SEAL and downloads the preprocessed datasets. To use it, execute the following commands:

```bash
docker build -t cryptotl .
docker run -it cryptotl
```

Then proceed with the CryptoTL commands stated above. For using Intel HEXL, modify ```Dockerfile``` accordingly.

### Citing our work

Please use the following BibTeX entry to cite our work in academic papers.

```tex
@article{DBLP:journals/corr/abs-2205-11935,
  author    = {Roman Walch and
               Samuel Sousa and
               Lukas Helminger and
               Stefanie N. Lindstaedt and
               Christian Rechberger and
               Andreas Tr{\"{u}}gler},
  title     = {CryptoTL: Private, efficient and secure transfer learning},
  journal   = {CoRR},
  volume    = {abs/2205.11935},
  year      = {2022}
}
```

FROM debian:11

ENV DEBIAN_FRONTEND noninteractive

# install dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake git python3 python3-pip wget

# copy
COPY . /build
WORKDIR /build

# get datasets
RUN cd datasets && \
    wget https://github.com/IAIK/CryptoTL/releases/download/v1.0.0/IMDB-test-SBERT.csv && \
    wget https://github.com/IAIK/CryptoTL/releases/download/v1.0.0/IMDB-train-SBERT.csv && \
    wget https://github.com/IAIK/CryptoTL/releases/download/v1.0.0/SBERT-youtube-combined.csv && \
    wget https://github.com/IAIK/CryptoTL/releases/download/v1.0.0/Twitter-test-SBERT.csv && \
    wget https://github.com/IAIK/CryptoTL/releases/download/v1.0.0/Twitter-train-SBERT.csv && \
    cd ..

RUN pip3 install -r requirements.txt

# compile SEAL:
RUN cd cpp && \
    rm -rf SEAL && \
    git clone https://github.com/microsoft/SEAL.git && \
    cd SEAL && \
    git checkout 4.0.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 4 && \
    cd ../../..

CMD ["/bin/bash"]

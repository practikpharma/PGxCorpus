# PGxCorpus named entity recognition system

Quick description of the system.

## Usage

### Downloading the PGxCorpus

cd data/

mkdir PGxCorpus

cd PGxCorpus

wget https://github.com/practikpharma/PGxCorpus/raw/master/PGxCorpus.tar

tar xvf PGxCorpus.tar

wget https://github.com/practikpharma/PGxCorpus/raw/master/PGxCorpus_pubtator.tar

tar xvf PGxCorpus_pubtator.tar

cd ../..

### Downloading and installing the lattice module for torch

git clone https://gitlab.inria.fr/jolegran/lattice

make install

### Running the experiment

th train.lua 


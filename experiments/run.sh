#!/bin/bash

CORPUSDB=data/million_post_corpus/corpus.sqlite3
DLURL=https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2

set -e
set -o pipefail

if [ ! -e "$CORPUSDB" ] ; then
    echo "Database file '$CORPUSDB' not found."
    echo "It is available at $DLURL"
    read -p "Do you want to download it? (y/n) " yn
    if [ "$yn" != 'y' ] ; then
        exit 1
    fi
    mkdir -p data
    if which wget > /dev/null ; then
        wget -q -O- "$DLURL" | tar -xj -C data
    elif which curl > /dev/null ; then
        curl -L "$DLURL" | tar -xj -C data
    else
        echo 'You need to install either wget or curl to download the data.'
        exit 1
    fi
fi

export PYTHONHASHSEED=123456
mkdir -p logs models plots tables
nice python src/train_word2vec.py   2>&1 | tee logs/word2vec_log.txt
nice python src/bocid_clustering.py 2>&1 | tee logs/bocid_clustering_log.txt
nice python src/train_doc2vec.py    2>&1 | tee logs/doc2vec_log.txt
nice python src/run_evaluation.py   2>&1 | tee logs/evaluation_log.txt
python src/result_plots_and_tables.py

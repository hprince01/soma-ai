#!/bin/sh
# Downloads the KINNEWS dataset (Niyongabo et al., COLING 2020, MIT licence) into data/raw/
set -e
mkdir -p data/raw && cd data/raw
curl -L -o KINNEWS.zip "https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"
unzip -oq KINNEWS.zip
echo "Done: data/raw/KINNEWS/raw/{train,test}.csv"

#/bin/bash
mkdir -p data

curl https://klejbenchmark.com/static/data/klej_cbd.zip --output ./data/klej_cbd.zip
unzip ./data/klej_cbd.zip
mv test_features.tsv data
mv train.tsv data
rm ./data/klej_cbd.zip

mkdir -p weights

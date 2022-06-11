mkdir -p data
mkdir -p data/cc100
wget -c "https://data.statmt.org/cc-100/pl.txt.xz" -O - | xz --decompress > data/cc100/pl.txt

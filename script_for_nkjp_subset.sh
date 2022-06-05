mkdir -p data
mkdir -p data/nkjp
wget -c "http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz" -O - | tar -xz -c data/nkjp

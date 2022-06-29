# KLEJ dataset downloading

## NKJP-NER: The task is to predict the type of the named entity.
## PolEmo2.0-IN: https://klejbenchmark.com/static/data/klej_polemo2.0-in.zip
## PolEmo2.0-OUT: https://klejbenchmark.com/static/data/klej_polemo2.0-out.zip

for klej_task in  klej_nkjp-ner klej_polemo2.0-in klej_polemo2.0-out
do
  mkdir -p datasets/${klej_task}
  curl https://klejbenchmark.com/static/data/${klej_task}.zip --output ./datasets/${klej_task}/${klej_task}.zip
  unzip ./datasets/${klej_task}/${klej_task}.zip
  mv test_features.tsv datasets/${klej_task}
  mv train.tsv datasets/${klej_task}
  mv dev.tsv datasets/${klej_task}
  rm ./datasets/${klej_task}/${klej_task}.zip
done







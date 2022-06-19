# DistilHerBERT

## Datasets

### CC100

- 40GB
- download to `data/cc100/pl.txt` using `cc100.sh`
- run `dataset.py`'s main to split downloaded dataset into `data/cc100/pl_train.txt`, `data/cc100/pl_test.txt`
  , `data/cc100/pl_valid.txt`
- `dataset.get_cc100_dataloader` will create trainloader, testloader and validloader but use it only after splitting
- `cc100_train.py` launches training on `cc100` dataset, but there's a couple of things that **must** be fixed
- the dataset is too big to be kept in memory, so it's streamed. This means shuffling will be a little more complex
  (not implemented for now)
- masking happens during data collation (it's the process of loading data into batches after tokenization)

https://app.neptune.ai/rm360179/DistilHerBERT

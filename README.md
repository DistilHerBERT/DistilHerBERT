# DistilHerBERT

## Datasets

### CC100
- 40GB
- download to `data/cc100/pl.txt` using `cc100.sh`
- `dataset.get_cc100_dataloader` will create a dataloader covering the whole dataset
- it is supposed to mask some (`mlm_prob`) of the input tokens, but not sure whether that works yet
- `cc100_train.py` launches training on `cc100` dataset, but it's for testing purpose for now (there's no actual
training, just checking whether the model will forward the data)
- the dataset is too big to be kept in memory, so it's streamed. This means shuffling will be a little more complex
(not implemented for now)
- data loading is not optimized yet

https://app.neptune.ai/rm360179/DistilHerBERT

import os.path

import datasets

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import torch


class CC100Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: datasets.iterable_dataset.IterableDataset):
        super(CC100Dataset).__init__()
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


def get_cc100_dataloader(tokenizer=AutoTokenizer.from_pretrained("allegro/herbert-base-cased"),
                         dataset: datasets.iterable_dataset.IterableDataset = None,
                         batch_size=10,
                         mlm_prob=0.15):
    if dataset is None:
        dataset = load_dataset("text",
                               data_files=os.path.join("data", "cc100", "pl.txt"),
                               streaming=True) \
            .map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", return_tensors='pt'),
                 batched=True).remove_columns("text")

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=mlm_prob)  # TODO: it should be sufficient to specify the tokenizer in just one place, not both in dataset and data collator...
    train = dataset['train']
    torch_train = CC100Dataset(train)
    train_loader = DataLoader(torch_train, batch_size=batch_size, collate_fn=data_collator)

    return train_loader

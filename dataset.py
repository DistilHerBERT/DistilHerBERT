import os.path
from zlib import crc32

import datasets

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import torch
from tqdm import tqdm
from neattext.functions import clean_text


class CC100Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: datasets.iterable_dataset.IterableDataset):
        super(CC100Dataset).__init__()
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2 ** 32


def str_to_float(s, encoding="utf-8"):
    return bytes_to_float(s.encode(encoding))

def imperfect_split_file_lines(file_path: str = os.path.join("data", "cc100", "pl.txt"),
                               train_fraction=0.8,
                               test_fraction=0.1,
                               valid_fraction=0.1,
                               train_output_file='data/cc100/pl_train.txt',
                               test_output_file='data/cc100/pl_test.txt',
                               valid_output_file='data/cc100/pl_valid.txt',
                               clean=False
                               ):
    with open(file_path, mode='r', encoding='utf-8') as read_file:
        with open(train_output_file, mode='w', encoding='utf-8') as train_file:
            with open(test_output_file, mode='w', encoding='utf-8') as test_file:
                with open(valid_output_file, mode='w', encoding='utf-8' ) as valid_file:
                    for _, line in enumerate(tqdm(read_file)):
                        sentence = line if not clean else clean_text(line)
                        num_between_0_and_1 = str_to_float(sentence)
                        if num_between_0_and_1 <= train_fraction:
                            train_file.write(sentence)
                        elif num_between_0_and_1 <= train_fraction + test_fraction:
                            test_file.write(sentence)
                        else:
                            valid_file.write(sentence)


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
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=mlm_prob)  # TODO: it should be sufficient to specify the tokenizer in just one place, not both in dataset and data collator...

    train = dataset['train']
    torch_train = CC100Dataset(train)
    train_loader = DataLoader(torch_train, batch_size=batch_size, collate_fn=data_collator)

    for i, _ in enumerate(tqdm(dataset['train'])):
        print(i)

    return train_loader


def main():
    imperfect_split_file_lines()


if __name__ == '__main__':
    main()

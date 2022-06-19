import os.path
from typing import Tuple, Iterable, Dict
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
                with open(valid_output_file, mode='w', encoding='utf-8') as valid_file:
                    for _, line in enumerate(tqdm(read_file)):
                        sentence = line if not clean else clean_text(line)
                        num_between_0_and_1 = str_to_float(sentence)
                        if num_between_0_and_1 <= train_fraction:
                            train_file.write(sentence)
                        elif num_between_0_and_1 <= train_fraction + test_fraction:
                            test_file.write(sentence)
                        else:
                            valid_file.write(sentence)


def create_collate_fn(tokenizer, mlm=True, mlm_prob=0.15, return_tensors='pt'):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=mlm,
                                                    mlm_probability=mlm_prob,
                                                    return_tensors=return_tensors)  # TODO: it should be sufficient to specify the tokenizer in just one place, not both in dataset an

    def collate(batch: Iterable[Dict]):
        texts = [x.pop('text') for x in batch]
        outcomes = data_collator(batch)
        for text, x in zip(texts, outcomes):
            x['text'] = text
        return outcomes

    return data_collator


def copy_input_ids(dict_: Dict, key='input_ids', newkey='unmasked_input_ids'):
    dict_[newkey] = dict_[key].detach().clone()
    return dict_


def get_cc100_dataloaders(tokenizer=AutoTokenizer.from_pretrained("allegro/herbert-base-cased"),
                          dir_path: str = os.path.join('data', 'cc100'),
                          batch_size=10,
                          mlm_prob=0.15) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    datasets = load_dataset("text", data_dir=os.path.join(dir_path), streaming=True).map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", return_tensors='pt'),
        batched=True).remove_columns("text").map(copy_input_ids)

    loaded_datasets = datasets.get('train'), datasets.get('test'), datasets.get('validation')
    loaded_datasets = filter(lambda x: x is not None, loaded_datasets)

    instantiated_datasets = [CC100Dataset(x) for x in loaded_datasets]
    return tuple(DataLoader(dataset, batch_size=batch_size,
                            collate_fn=create_collate_fn(tokenizer, mlm=True, mlm_prob=mlm_prob, return_tensors='pt'))
                 for dataset in
                 instantiated_datasets)


def main():
    imperfect_split_file_lines()


if __name__ == '__main__':
    main()

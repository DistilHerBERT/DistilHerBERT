from torch.utils.data import Dataset, DataLoader
import csv
import torch
from transformers import AutoTokenizer, AutoModel


class CBDDataset(Dataset):
    def __init__(self):
        texts = list()
        labels = list()
        with open('data/train.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            # to skip column names
            next(reader)

            for text, label in reader:
                texts.append(text)
                labels.append(int(label))

        self.texts = texts
        self.labels = labels


    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.texts)

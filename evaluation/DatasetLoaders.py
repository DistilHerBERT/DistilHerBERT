from neattext.functions import clean_text
from torch.utils.data import Dataset
import torch
import csv


class KlejDataset(Dataset):
    """
    Dataloader for dataset from KLEJ for the following task :  NKJP_NER, POLEMO2.0-in, POLEMO2.0-out
    """

    def __init__(self, path, tokenizer, device, labels_map=None):
        texts = list()
        labels = list()
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            # to skip column names
            next(reader)

            for utterance, target in reader:
                texts.append(clean_text(utterance))
                labels.append(target)

        self.texts = texts
        self.tokenizer = tokenizer
        self.device = device

        self.labels_map = labels_map if labels_map is not None else {label: i for i, label in enumerate(set(labels), 0)}
        self.labels_no = len(self.labels_map)
        self.labels = [self.labels_map[label] for label in labels]

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.texts[idx], truncation=True, padding="max_length")
        item = {key: torch.tensor(tokenized.data[key], device=self.device) for key in tokenized.data}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        return item

    def __len__(self):
        return len(self.texts)


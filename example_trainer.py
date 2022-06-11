import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# example dataset
class CustomImageDataset(Dataset):
    def __init__(self, sentences_path):
        self.sentences = pd.read_csv(sentences_path)['sentence'].values

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        data = {'sentence': self.sentences[idx]}
        return data

dataset = CustomImageDataset('datasets/datasetSentences.csv')

dataloader = DataLoader(dataset, batch_size=32)
loaders  = {'train': dataloader, 'test': dataloader}

from trainer.utils import get_teacher_student_tokenizer
teacher, student, tokenizer = get_teacher_student_tokenizer()

from trainer.distilTrainer import DistilTrainer

params_trainer = {
    'teacher': teacher.to(device),
    'student': student.to(device),
    'tokenizer': tokenizer,
    'loaders': loaders,
    'criterion1': nn.CrossEntropyLoss().to(device),
    # 'criterion2': nn.CrossEntropyLoss().to(device),
    'criterion2': nn.KLDivLoss('batchmean', log_target=True).to(device), # mam używać log_target?
    'criterion3': nn.CosineEmbeddingLoss().to(device),
    'optim': torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0), # wyrzucić z wd embedingi i batchnormalization
    'device': device
}
trainer = DistilTrainer(**params_trainer)

params_run = {
    'epoch_start': 0,
    'epoch_end': 2,
    'exp_name': 'plain_distil',
    'save_interval': 100,
    'random_seed': 42
}

trainer.run_exp(**params_run)
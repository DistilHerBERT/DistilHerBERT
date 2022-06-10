from transformers import AutoTokenizer, AutoModel
from models.architectures import BertCBD
import torch.optim as optim
from datasets import load_dataset
from torch import nn
import torch
from logger.logger import NeptuneLogger
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from local_datasets import CBDDataset



hgbert = AutoModel.from_pretrained("allegro/herbert-base-cased")

model = BertCBD(hgbert.config)

#model.load_state_dict(torch.load('weights/cbd_net10.pth', map_location=torch.device('cpu')))
model.load_state_dict(torch.load('weights/cbd_net11.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

def run_once(x):
    x = [x]
    tokenized = tokenizer.batch_encode_plus(x, padding="longest", add_special_tokens=True, return_tensors='pt')
    tokenized.pop('token_type_ids')
    output = model(**tokenized)
    return output

ds = CBDDataset('data/train.tsv')
print(len(ds))
#asdf = asdf
pos = 0
true_pos = 0
for i in range(300):
    item = ds[i]
    x = item[0]
    label = item[1]
    print(x)
    print(label)
    #print((run_once(x) > 0)*1.0)
    oo =run_once(x)
    print(oo)
    if label == 1:
        pos += 1
        if oo > 0:
            true_pos += 1
print(true_pos / pos)
while True:
    x = input()
    print(run_once(x))
'''
zeros, ones = 0, 0
for i in range(len(ds)):
    if ds[i][1] == 0:
        zeros += 1
    else:
        ones += 1

print(zeros, ones)
'''

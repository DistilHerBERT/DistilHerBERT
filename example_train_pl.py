#from models.architectures import BertAgNews
from models.architectures import BertCBD
import torch.optim as optim
from datasets import load_dataset
from torch import nn
import torch
from logger.logger import NeptuneLogger
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from local_datasets import CBDDataset

training_data = CBDDataset()
#print(train)


epochs = 10
save_path = './weights/agnews_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_interval = 100
lr = 0.001
batch_size = 32


log = NeptuneLogger()
log['lr'] = lr
log['epochs'] = epochs

# loading data train = test on purpose, this is just an example
#dataset = load_dataset("ag_news")

#train = dataset['train']
#test = dataset['test']


# prepare the net
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased")

'''
train = train.shuffle()
train = train.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
#print(len(train))
# copyw

'''

#training_data = TextDataset()
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


#net = BertAgNews(model_pl.config)
net = BertCBD(model_pl.config)
net.bert.load_state_dict(model_pl.state_dict())



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

net.train()
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    log['epoch'].log(epoch)
    for i, data in enumerate(train_dataloader):
        log['step'].log(i)
        x, y = data
        x = list(x)
        x = tokenizer.batch_encode_plus(x, padding="longest", add_special_tokens=True, return_tensors='pt')
        y = torch.tensor(y)
        #print(y)

        x.pop('token_type_ids')


        #input_ids = data['input_ids']
        #jiprint(input_ids)
        #jattention_mask = data['attention_mask']
        #label = data['label']
        # label denotes news type, 0, 1, 2, 3 - "World", “Sports”, “Business”, “Sci/Tech

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(**x
            #input_ids=input_ids,
            #attention_mask=attention_mask
        )
        outputs = torch.nn.Softmax(dim=1)(outputs)
        #print(outputs)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            tmp_loss = running_loss / 100
            log['train_loss'].log(tmp_loss)
            running_loss = 0.0

        if i % save_interval == save_interval - 1:
            torch.save(net.state_dict(), save_path)

'''
test = test.shuffle()
test = test.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

net.eval()
with torch.no_grad():
    for i, data in enumerate(test):

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        label = data['label']

        outputs = net(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        outputs = torch.nn.Softmax(dim=1)(outputs)

        loss = criterion(outputs, label)
        log['test_loss'].log(tmp_loss)
'''

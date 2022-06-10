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

data = CBDDataset('data/train.tsv')
training_data, validation_data = torch.utils.data.random_split(data, [9041, 1000])
#print(train)


epochs = 100
save_path = './weights/cbd_net11.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_interval = 100
lr = 0.000001 # 0.0001
batch_size = 32


log = NeptuneLogger()
log['lr'] = lr
log['epochs'] = epochs
log['device'] = device
log['batch_size'] = batch_size
log['weights'] = save_path


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
validate_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)


#net = BertAgNews(model_pl.config)
net = BertCBD(model_pl.config)
net.bert.load_state_dict(model_pl.state_dict())
net.to(device)



optimizer = optim.Adam(net.parameters(), lr=lr)

net.train()

for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
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
        x = x.to(device)
        y = y.to(device)


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
        #outputs = torch.nn.Softmax(dim=1)(outputs)
        #print(outputs)

        w = torch.ones(outputs.reshape(-1).shape).cuda() * 10.
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=w)
        loss = criterion(outputs.reshape(-1), y*1.0)
        loss.backward()
        optimizer.step()
        log['step_loss'].log(loss.item())

        # print statistics
        running_loss += loss.item()
        if i % save_interval == save_interval-1:    # print every 2000 mini-batches
            torch.save(net.state_dict(), save_path)
            tmp_loss = running_loss / save_interval
            log['train_loss'].log(tmp_loss)
            running_loss = 0.0

            #evaluate
            net.eval()
            with torch.no_grad():
                val_loss = 0.
                val_loss_count = 0
                correct = 0.
                all_ = 0.

                for i, data in enumerate(validate_dataloader):
                    x, y = data
                    x = list(x)
                    x = tokenizer.batch_encode_plus(x, padding="longest", add_special_tokens=True, return_tensors='pt')
                    y = torch.tensor(y)
                    #print(y)

                    x.pop('token_type_ids')
                    x = x.to(device)
                    y = y.to(device)


                    outputs = net(**x
                        #input_ids=input_ids,
                        #attention_mask=attention_mask
                    )
                    correct += ((outputs.reshape(-1) > 0) * 1.0 == y).sum()
                    #print(outputs.reshape(-1).to('cpu'))
                    #print(y.to('cpu'))
                    all_ += len(outputs)
                    w = torch.ones(outputs.reshape(-1).shape).cuda() * 10.
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=w)
                    print(outputs.reshape(-1).cpu())
                    print(y.reshape(-1).cpu() * 1.0)
                    loss = criterion(outputs.reshape(-1), y*1.0)
                    val_loss += loss.item()
                    val_loss_count += 1
                log['val_loss'].log(val_loss/val_loss_count)
                log['val_acc'].log(correct/all_)

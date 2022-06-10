import datasets.iterable_dataset

from models.architectures import BertAgNews
import torch.optim as optim
from datasets import load_dataset
from torch import nn
import torch
from logger.logger import NeptuneLogger, DummyLogger

from transformers import AutoTokenizer, AutoModel
from torchtext.datasets import CC100


class CC100Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: datasets.iterable_dataset.IterableDataset):
        super(CC100Dataset).__init__()
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


if __name__ == '__main__':
    epochs = 10
    # save_path = './agnews_net.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_interval = 100
    lr = 0.0001
    batch_size = 5

    log = NeptuneLogger()
    log['lr'] = lr
    log['epochs'] = epochs
    log['device'] = device
    log['batch_size'] = batch_size

    # loading data train = test on purpose, this is just an example
    # dataset = load_dataset("ag_news")

    # train = dataset['test']
    # test = dataset['test']

    # cc100_dataset = CC100("data/cc100/pl", language_code='pl')

    # prepare the net
    # cc100_dataset = load_dataset("text", data_files="https://data.statmt.org/cc-100/pl.txt.xz")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    cc100_dataset = load_dataset("text",
                                 data_files="data\\cc100\\pl1.txt", # TODO: change dataset path...
                                 streaming=True) \
        .map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)
    train = cc100_dataset['train']
    model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased")

    train = train.shuffle() # TODO: this shuffles only buffer_size elements, by default 1k
    # TODO: add train-test split
    torch_train = CC100Dataset(train)
    # train = train.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
    # train.set_format()
    # train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(torch_train, batch_size=batch_size)
    # print(len(train))
    # copyw
    net = BertAgNews(model_pl.config)
    net.bert.load_state_dict(model_pl.state_dict())
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
    log['epoch'].log(epoch)
    for i, data in enumerate(train_loader):
        log['step'].log(i)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    label = data['label']
    # label denotes news type, 0, 1, 2, 3 - "World", “Sports”, “Business”, “Sci/Tech
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    label = label.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    outputs = torch.nn.Softmax(dim=1)(outputs)

    loss = criterion(outputs, label)
    log['train_every_step'].log(loss)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 100 == 99:  # print every 2000 mini-batches
        tmp_loss = running_loss / 100
    log['train_loss'].log(tmp_loss)
    running_loss = 0.0

    # if i % save_interval == save_interval - 1:
    #     torch.save(net.state_dict(), save_path)

    # test = test.shuffle()
    # test = test.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
    # test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    net.eval()
    # with torch.no_grad():
    #     for i, data in enumerate(test):
    #     input_ids = data['input_ids']
    #     attention_mask = data['attention_mask']
    #     label = data['label']
    #
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    #     label = label.to(device)
    #
    #     outputs = net(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask
    #     )
    #     outputs = torch.nn.Softmax(dim=1)(outputs)
    #
    #     loss = criterion(outputs, label)
    #     log['test_loss'].log(tmp_loss)

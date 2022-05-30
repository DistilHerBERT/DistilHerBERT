from models.architectures import BertAgNews
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torch import nn
import torch
from logger.logger import NeptuneLogger

epochs = 10
save_path = './weights/agnews_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_interval = 10_000
lr = 0.001


log = NeptuneLogger()
log['lr'] = lr
log['epochs'] = epochs

# loading data train = test on purpose, this is just a template
train_iter = AG_NEWS(split='test')
test_iter = AG_NEWS(split='test')

# prepare the net
net = BertAgNews()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_iter):
        # label denotes news type, 0, 1, 2, 3 - "World", “Sports”, “Business”, “Sci/Tech
        label, line = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(line)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            tmp_loss = running_loss / 2000
            log['train_loss'].log(tmp_loss)
            running_loss = 0.0

        if i % save_interval == save_interval - 1:
            torch.save(net.state_dict(), save_path)


net.eval()
with torch.no_grad():
    for i, data in enumerate(test_iter):
        label, line = data
        outputs = net(line)
        loss = criterion(outputs, label)
        log['test_loss'].log(loss)

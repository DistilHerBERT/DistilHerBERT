from dataset import get_cc100_dataloaders
import torch.optim as optim
from torch import nn
import torch
from logger.logger import NeptuneLogger

from transformers import AutoModel

from models.architectures import BertForMLM


def main():
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_interval = 100
    lr = 0.0001
    batch_size = 5
    save_path = './cc100_mlm_net.pth'

    log = NeptuneLogger()
    log['lr'] = lr
    log['epochs'] = epochs
    log['device'] = device
    log['batch_size'] = batch_size

    # prepare the net
    model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased")

    train_loader, test_loader, valid_loader = get_cc100_dataloaders(mlm_prob=0.15)
    net = BertForMLM(model_pl.config)
    # TODO: what about other parts of net? (LM head?) no loading?
    net.bert.load_state_dict(model_pl.state_dict())
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        log['epoch'].log(epoch)
        size = 0
        for i, data in enumerate(train_loader):
            size += data['input_ids'].shape[0]
            # log['step'].log(i)
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['unmasked_input_ids']

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            log['train_every_step'].log(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                tmp_loss = running_loss / 100
                log['train_loss'].log(tmp_loss)
                running_loss = 0.0

            if i % save_interval == save_interval - 1:
                torch.save(net.state_dict(), save_path)

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


if __name__ == '__main__':
    main()

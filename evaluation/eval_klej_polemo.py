from evaluation.DatasetLoaders import KlejDataset
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from logger.logger import NeptuneLogger

from transformers import AutoTokenizer, AutoModel

from models.distil_student import creat_student
from models.klej.bert_polemo import BertPolemo

torch.cuda.empty_cache()
epochs = 20
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
print(torch.cuda.is_available())
save_interval = 200
lr = 0.0001
batch_size = 4


def get_dataloaders(tokenizer, path_train, path_test):
    train_set = KlejDataset(path_train, tokenizer, device)
    print(train_set.labels_map)
    test_set = KlejDataset(path_test, tokenizer, device, train_set.labels_map)
    labels = train_set.labels_map
    train = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    test = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size)

    return train, test, labels


def score_model(output, target):
    # accuracy
    _, preds = torch.max(output, dim=1)
    acc = torch.sum(preds == target)
    return acc


def main(tokenizer, model, save_path, log, dataset_train_path, dataset_test_path ):
    train, test, labels_map = get_dataloaders(tokenizer, dataset_train_path,
                                              dataset_test_path)

    model_polemo = BertPolemo(model.config, len(labels_map))
    model_polemo.bert.load_state_dict(model.state_dict(), strict=False)
    model_polemo.to(device)

    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model_polemo.train()
        running_loss, running_acc, final_acc = 0.0, 0.0, 0.0
        log['epoch'].log(epoch)
        for i, data in enumerate(train):
            log['step'].log(i)
            optimizer.zero_grad()
            input_ids, attention_mask, label = data['input_ids'], data['attention_mask'], data['label']
            outputs = model_polemo(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc = score_model(outputs, label)
            running_acc += acc
            final_acc += acc
            if i % 10 == 9:
                tmp_loss = running_loss / 10
                log['train_loss'].log(tmp_loss)
                log['train_acc'].log(running_acc / batch_size / 10)
                running_loss, running_acc = 0.0, 0.0

            if i % save_interval == save_interval - 1:
                torch.save(model_polemo.state_dict(), save_path)
        log['epoch_train_acc'].log(final_acc / len(train))

        model_polemo.eval()
        running_loss_test, running_acc_test, final_acc_test = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(test):
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                label = data['label']

                outputs = model_polemo(input_ids=input_ids, attention_mask=attention_mask)
                outputs = torch.nn.Softmax(dim=1)(outputs)

                loss = criterion(outputs, label)
                running_loss_test += loss.item()
                acc = score_model(outputs, label)
                running_acc_test += acc
                final_acc_test += acc
                if i % 10 == 9:
                    tmp_loss = running_loss / 10
                    log['test_loss'].log(tmp_loss)
                    log['test_acc'].log(running_acc_test / batch_size / 10)
                    running_loss_test, running_acc_test = 0.0, 0.0
            log['epoch_test_acc'].log(final_acc_test / len(test))


def run_scenario(case, dataset):
    '''

    :param case: '0' for herbert with half layers, '1' for HerBERT, '2' for distiled HerBERT
    :param dataset: 'in' or 'out'
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased", return_dict=False).to(device=device)

    if case == "0":
        student = creat_student(model_pl)
        path = f'./weights/klej_polemo_{dataset}_herbert_2.pth'
        log = NeptuneLogger(f"HerBert_polemo_{dataset}")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_polemo2.0-{dataset}'
        log['model_name'] = 'student_2'
        main(tokenizer, student, path, log, f"datasets/klej_polemo2.0-{dataset}/train.tsv", f"datasets/klej_polemo2.0-{dataset}/dev.tsv")
    elif case == "1":
        path = f'./weights/klej_polemo_{dataset}_herbert.pth'
        log = NeptuneLogger(f"HerBert_polemo_{dataset}")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_polemo2.0-{dataset}'
        log['model_name'] = 'herbert'
        main(tokenizer, model_pl, path, log, f"datasets/klej_polemo2.0-{dataset}/train.tsv",
             f"datasets/klej_polemo2.0-{dataset}/dev.tsv")
    elif case == "2":
        distil_path = './weights/student_final.pth'
        model = creat_student().to(device=device)
        model.load_state_dict(torch.load(distil_path, map_location=device))
        path = f'./weights/klej_polemo_{dataset}_herbert_distil.pth'
        log = NeptuneLogger(f"HerBert_polemo_{dataset}_herbert_distil")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_polemo2.0-{dataset}'
        log['model_name'] = 'student_distil'
        main(tokenizer, model, path, log, f"datasets/klej_polemo2.0-{dataset}/train.tsv",
             f"datasets/klej_polemo2.0-{dataset}/dev.tsv")


if __name__ == "__main__":
    run_scenario('2', 'in')

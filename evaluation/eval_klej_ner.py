from evaluation.DatasetLoaders import KlejDataset
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch
from logger.logger import NeptuneLogger

from transformers import AutoTokenizer, AutoModel

from models.distil_student import creat_student
from models.klej.bert_ner import BertNER

torch.cuda.empty_cache()
epochs = 20
save_path = './weights/klej_ner_herbert.pth'
device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps')
print(torch.cuda.is_available())
save_interval = 100
lr = 0.00001
batch_size = 8

log = NeptuneLogger("HerBert_NER")
log['lr'] = lr
log['epochs'] = epochs


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
    acc = (output.argmax(dim=1) == target).sum().item()
    return acc


def main(tokenizer, model, save_path, log, dataset_train_path="datasets/klej_nkjp-ner/train.tsv",
         dataset_test_path="datasets/klej_nkjp-ner/dev.tsv"):
    train, test, labels_map = get_dataloaders(tokenizer, dataset_train_path, dataset_test_path)

    ner_model = BertNER(model.config, len(labels_map))
    ner_model.bert.load_state_dict(model.state_dict(), strict=False)
    ner_model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    optimizer = optim.SGD(ner_model.parameters(), lr=lr, momentum=0.9)
    running_loss, running_acc, final_acc = 0.0, 0.0, 0.0
    for epoch in range(epochs):
        ner_model = ner_model.train()
        final_acc = 0.0
        log['epoch'].log(epoch)
        for i, data in enumerate(train):
            log['step'].log(i)

            input_ids, attention_mask, label = data['input_ids'], data['attention_mask'], data['label']
            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(ner_model.parameters(), max_norm=1.0)
            optimizer.zero_grad()

            running_loss += loss.item()
            acc = score_model(outputs, label)
            running_acc += acc
            final_acc += acc
            if i % save_interval == 99:
                tmp_loss = running_loss / save_interval / batch_size
                log['train_loss'].log(tmp_loss)
                log['train_acc'].log(running_acc / batch_size / save_interval)
                running_loss, running_acc = 0.0, 0.0

            if i % save_interval == save_interval - 1:
                torch.save(ner_model.state_dict(), save_path)
        log['epoch_train_acc'].log(final_acc / len(train.dataset))
        torch.save(ner_model.state_dict(), save_path)
        ner_model = ner_model.eval()
        running_loss_test, running_acc_test, final_acc_test = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(test):
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                label = data['label']

                outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, label)
                running_loss_test += loss.item()
                acc = score_model(outputs, label)
                running_acc_test += acc
                final_acc_test += acc
                if i % save_interval == 99:
                    tmp_loss = running_loss_test / save_interval / batch_size
                    log['test_loss'].log(tmp_loss)
                    log['test_acc'].log(running_acc_test / batch_size / save_interval)
                    running_loss_test, running_acc_test = 0.0, 0.0
            log['test_acc'].log(final_acc_test / len(test.dataset))


def run_scenario(case):
    '''

    :param case: '0' for herbert with half layers, '1' for HerBERT, '2' for distiled HerBERT
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased", return_dict=False).to(device=device)

    if case == "0":
        student = creat_student(model_pl)
        path = './weights/klej_ner_herbert_2.pth'
        log = NeptuneLogger(f"HerBert_ner")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'student_2'
        main(tokenizer, student, path, log)
    elif case == "1":
        path = f'./weights/klej_ner_herbert.pth'
        log = NeptuneLogger(f"HerBert_ner")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'herbert'
        main(tokenizer, model_pl, path, log)
    elif case == "2":
        distil_path = './weights/student_final.pth'
        model = creat_student()
        model.load_state_dict(torch.load(distil_path, map_location=device))
        path = f'./weights/klej_ner_herbert_distil.pth'
        log = NeptuneLogger(f"HerBert_ner_herbert_distil")
        log['lr'] = lr
        log['epochs'] = epochs
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'student_distil'
        main(tokenizer, model, path, log)


if __name__ == "__main__":
    run_scenario('2')
    run_scenario('0')
    run_scenario('1')

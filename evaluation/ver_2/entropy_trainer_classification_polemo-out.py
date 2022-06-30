import torch
from accelerate import Accelerator
from torch import nn

# init accelerator
accelerator = Accelerator(device_placement=True, fp16=True, mixed_precision='fp16')
device = accelerator.device

EPOCHS = 8
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 64 // BATCH_SIZE
LR = 2e-5


from torch.utils.data import DataLoader
from evaluation.DatasetLoaders import KlejDataset
from transformers import AutoTokenizer


def get_dataloaders(tokenizer, path_train, path_test):
    train_set = KlejDataset(path_train, tokenizer, device)
    print(train_set.labels_map)
    test_set = KlejDataset(path_test, tokenizer, device, train_set.labels_map)
    labels = train_set.labels_map
    train = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE)
    test = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE)

    return train, test, labels

import collections
from models.klej.bert_polemo import BertPolemo
from models.distil_student import creat_student
from transformers import AdamW, get_cosine_schedule_with_warmup
from trainers.utils import configure_optimizer
from trainers.vanillaTrainerClassifier import VanillaTrainerClassifier

def main(tokenizer, model, save_path, log, case):
    dataset_train_path = "datasets/klej_polemo2.0-out/train.tsv"
    dataset_test_path = "datasets/klej_polemo2.0-out/dev.tsv"
    train_loader, test_loader, labels_map = get_dataloaders(tokenizer, dataset_train_path, dataset_test_path)

    polemo_model = BertPolemo(model.config, len(labels_map))
    polemo_model.bert.load_state_dict(model.state_dict(), strict=False)
    polemo_model.to(device)

    # set accelerator

    optim = configure_optimizer(polemo_model, AdamW, weight_decay=1e-3, lr=LR)

    # TU ZMIENIŁEM
    train_loader, test_loader, polemo_model, optim = accelerator.prepare(
        train_loader, test_loader, polemo_model, optim)

    loaders = {'train': train_loader, 'test': test_loader}

    NUM_TRAINING_STEPS = len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_cycles=EPOCHS,
        num_warmup_steps=int(0.01 * NUM_TRAINING_STEPS),
        num_training_steps=NUM_TRAINING_STEPS)

    params_trainer = {
        'model': polemo_model,  # .to(device),
        'tokenizer': tokenizer,
        'loaders': loaders,
        'criterion': nn.CrossEntropyLoss().to(device),
        'optim': optim,
        'scheduler': scheduler,
        'accelerator': accelerator,
        'device': device
    }
    trainer = VanillaTrainerClassifier(**params_trainer)
    config_run_epoch = collections.namedtuple('RE', ['save_interval', 'grad_accum_steps', 'running_step'])(20,
                                                                                                           GRAD_ACCUM_STEPS,
                                                                                                           40)
    params_run = {
        'epoch_start': 0,
        'epoch_end': EPOCHS,
        'exp_name': f'classification_polemo-in-case:{case}',
        'logger': log,
        'config_run_epoch': config_run_epoch,
        'random_seed': 42
    }

    trainer.run_exp(**params_run)



from logger.logger import NeptuneLogger
def run_scenario(case):
    '''

    :param case: '0' for herbert with half layers, '1' for HerBERT, '2' for distiled HerBERT
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    # model_pl = AutoModel.from_pretrained("allegro/herbert-base-cased", return_dict=False).to(device=device)

    if case == "0":
        distil_path = 'weights/plain_distil/2022-06-26_03-19-38/checkpoints/student_orginal_training.pth'
        model = creat_student().to(device=device)
        model.load_state_dict(torch.load(distil_path, map_location=device))
        path = 'weights/klej_ner_herbert_student_orginal_training.pth'
        log = NeptuneLogger(f"HerBert_ner_student_orginal_training")
        log['lr'] = LR
        log['epochs'] = EPOCHS
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'student_orginal_training'
        main(tokenizer, model, path, log, case)
    elif case == "1":
        distil_path = 'weights/plain_distil/2022-06-26_03-20-39/checkpoints/student_one_loss.pth'
        model = creat_student().to(device=device)
        model.load_state_dict(torch.load(distil_path, map_location=device))
        path = f'weights/klej_ner_herbert_student_one_loss.pth'
        log = NeptuneLogger(f"HerBert_ner_student_one_loss")
        log['lr'] = LR
        log['epochs'] = EPOCHS
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'student_one_loss'
        main(tokenizer, model, path, log, case)
    elif case == "2":
        distil_path = 'weights/plain_distil/2022-06-26_03-21-40/checkpoints/student_no_teacher.pth'
        model = creat_student().to(device=device)
        model.load_state_dict(torch.load(distil_path, map_location=device))
        path = f'weights/klej_ner_herbert_student_no_teacher.pth'
        log = NeptuneLogger(f"HerBert_ner_herbert_student_no_teacher")
        log['lr'] = LR
        log['epochs'] = EPOCHS
        log['dataset'] = f'klej_nkjp-ner'
        log['model_name'] = 'student_no_teacher'
        main(tokenizer, model, path, log, case)


if __name__ == "__main__":
    run_scenario('0')
    run_scenario('2')
    run_scenario('1')
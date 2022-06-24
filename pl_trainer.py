import os
import pandas as pd
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import collections
from trainers.utils import get_teacher_student_tokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer
import random
from trainers.distilTrainer import DistilTrainer

# init accelerator
accelerator = Accelerator(device_placement=True, fp16=True, mixed_precision='fp16')
device = accelerator.device

EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 200 // BATCH_SIZE

teacher, student, tokenizer = get_teacher_student_tokenizer()


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    input_labels, output_labels = zip(*data)
    lengths = [len(ids) for ids in input_labels]
    max_len = max(lengths)
    il_padded = []
    ol_padded = []
    for i in range(len(input_labels)):
        il = input_labels[i] + [1] * (max_len - len(input_labels[i]))
        ol = output_labels[i] + [1] * (max_len - len(output_labels[i]))
        il_padded.append(il)
        ol_padded.append(ol)
    
    il_padded = torch.tensor(il_padded, dtype=torch.long)
    ol_padded = torch.tensor(ol_padded, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return il_padded, ol_padded, lengths


# example dataset
class CustomImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, sentences_path, tokenizer, length):
        #self.sentences = pd.read_csv(sentences_path, sep=sep)['sentence']
        self.f = open(sentences_path)
        self.main_tokenizer = tokenizer
        self.aux_tokenizer = MosesTokenizer(lang='pl')
        self.vocab = self.main_tokenizer.vocab
        self.vocab_size = len(tokenizer.vocab)
        self.length = length


    def __len__(self):
        return self.length

    def __iter__(self):
        for line in self.f:
            input_labels, output_labels = self.random_word(line.strip())

            # [CLS] tag = BOS tag, [SEP] tag = SEP tag
            t1 = [self.main_tokenizer.bos_token_id] + input_labels + [self.main_tokenizer.sep_token_id]
            t1_label = [self.main_tokenizer.bos_token_id] + output_labels + [self.main_tokenizer.sep_token_id]

            bert_input = t1[:self.main_tokenizer.max_len_single_sentence]
            bert_label = t1_label[:self.main_tokenizer.max_len_single_sentence]

            yield bert_input, bert_label
    
    def random_word(self, sentence):
        tokens = self.aux_tokenizer.tokenize(sentence)
        input_labels = []
        output_labels = []

        for i, token in enumerate(tokens):
            bpe_tokens = self.main_tokenizer.tokenize(tokens[i])
            nb_bpe_tokens = len(bpe_tokens)
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                
                # 80% randomly change token to mask token
                if prob < 0.8:
                    mask_id = self.vocab['<mask>']
                    input_labels += [mask_id] * nb_bpe_tokens

                # 10% randomly change token to random token
                elif prob < 0.9:
                    for _ in range(nb_bpe_tokens):
                        random_id = random.randrange(self.vocab_size)
                        input_labels.append(random_id)

                # 10% randomly change token to current token
                else:
                    input_labels += [self.vocab[token] for token in bpe_tokens]

                output_labels += [self.vocab[token] for token in bpe_tokens]

            else:
                input_labels += [self.vocab[token] for token in bpe_tokens]
                output_labels += [self.main_tokenizer.pad_token_id] * nb_bpe_tokens
        
        return input_labels, output_labels


#TODO fix this
train_dataset = CustomImageDataset('datasets/pl_train.txt', tokenizer=tokenizer, length=4950457)
test_dataset = CustomImageDataset('datasets/pl_test.txt', tokenizer=tokenizer, length=110000)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, collate_fn=collate_fn)

# batch = next(iter(test_loader))
# batch

# set accelerator
from transformers import AdamW, get_cosine_schedule_with_warmup
from trainers.utils import configure_optimizer

# wyrzucić z wd embedingi i batchnormalization
# optim = configure_optimizer(student, AdamW, weight_decay=1e-3, lr=1e-4)
optim = AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-4, weight_decay=1e-3)

train_loader, test_loader, teacher, student, optim = accelerator.prepare(
    train_loader, test_loader, teacher, student, optim)

loaders  = {'train': train_loader, 'test': test_loader}

NUM_TRAINING_STEPS = len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim,
        num_cycles=EPOCHS,
        num_warmup_steps=int(0.15 * NUM_TRAINING_STEPS),
        num_training_steps=NUM_TRAINING_STEPS)


params_trainer = {
    'teacher': teacher,#.to(device),
    'student': student,#.to(device),
    'tokenizer': tokenizer,
    'loaders': loaders,
    'criterion1': nn.CrossEntropyLoss().to(device),
    'criterion2': nn.CrossEntropyLoss().to(device),
    # 'criterion2': nn.KLDivLoss('batchmean').to(device), # mam używać log_target?
    'criterion3': nn.CosineEmbeddingLoss().to(device),
    'optim': optim,
    'scheduler': scheduler,
    'accelerator': accelerator,
    'device': device
}
trainer = DistilTrainer(**params_trainer)

config_run_epoch = collections.namedtuple('RE', ['save_interval', 'grad_accum_steps', 'running_step'])(3_000, GRAD_ACCUM_STEPS, 30)

params_run = {
    'epoch_start': 0,
    'epoch_end': EPOCHS,
    'exp_name': f'plain_distil_scheduler:cosine,accelerate:bf16,batch_size:{BATCH_SIZE},hwmasking,acc_grad,cos_logits,clip_grad',
    'config_run_epoch': config_run_epoch,
    'temp': 2.,
    'random_seed': 42
}

trainer.run_exp(**params_run)
trainer.n_logger.run.stop()

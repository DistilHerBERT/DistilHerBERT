import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from sacremoses import MosesTokenizer


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
class CustomImageDataset(Dataset):
    def __init__(self, sentences_path, sep, tokenizer):
        self.sentences = pd.read_csv(sentences_path, sep=sep)['sentence']
        self.main_tokenizer = tokenizer
        self.aux_tokenizer = MosesTokenizer(lang='pl')
        self.vocab = self.main_tokenizer.vocab
        self.vocab_size = len(tokenizer.vocab)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_labels, output_labels = self.random_word(self.sentences[idx])

        # [CLS] tag = BOS tag, [SEP] tag = SEP tag
        t1 = [self.main_tokenizer.bos_token_id] + input_labels + [self.main_tokenizer.sep_token_id]
        t1_label = [self.main_tokenizer.bos_token_id] + output_labels + [self.main_tokenizer.sep_token_id]

        bert_input = t1[:self.main_tokenizer.max_len_single_sentence]
        bert_label = t1_label[:self.main_tokenizer.max_len_single_sentence]

        return bert_input, bert_label

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
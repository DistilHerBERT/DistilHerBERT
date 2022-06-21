import torch
import numpy as np

from models.distil_student import creat_student


# czy powinniśmy zasłaniać unk token?
def create_masked_ids(data):
    mask1 = torch.rand(data.input_ids.shape) < 0.15
    mask2 = torch.tensor(~np.isin(data.input_ids.detach().cpu().numpy(), (0, 1, 2)))
    masked_ids = (mask1 * mask2)
    data.input_ids[masked_ids] = 4
    return masked_ids

def get_masked_mask_and_att_mask(tokenized_input, lengths):
    mask1 = torch.tensor(~np.isin(tokenized_input.detach().cpu().numpy(), (0, 1, 2, 4)))
    att_mask = torch.arange(mask1.size(1))[None, :] < lengths[:, None].detach().cpu()
    return mask1, att_mask

def get_teacher_student_tokenizer():
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    teacher = AutoModel.from_pretrained("allegro/herbert-base-cased")
    student = creat_student(teacher)
    
    print(f'Number of parameters: Teacher: {count_parameters(teacher)}, Student: {count_parameters(student)},'
          f'Student / Teacher ratio: {round(count_parameters(student) / count_parameters(teacher), 4)}.')

    for params in teacher.parameters():
        params.requires_grad = False

    return teacher, student, tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_optimizer(model, optim, weight_decay=1e-4, **optim_kwargs):
    from torch import nn
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    # whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight'):
                if isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)


    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # inter_params = decay & no_decay
    # union_params = decay | no_decay
    # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    #                                             % (str(param_dict.keys() - union_params), )
    # print()
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = optim(optim_groups, **optim_kwargs)
    return optimizer

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
    alert_chunks = ['embeddings', 'LayerNorm', 'bias']
    no_decay = {pn for pn, p in model.named_parameters() if any(c in pn for c in alert_chunks)}
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in model.named_parameters() if pn not in no_decay and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for pn, p in model.named_parameters() if pn in no_decay and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer

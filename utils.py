import torch
import numpy as np

# czy powinniśmy zasłaniać unk token?
from projekt_nlp.DistilHerBERT.models.distil_student import creat_student


def create_masked_ids(data):
    mask1 = torch.rand(data.input_ids.shape) < 0.15
    mask2 = torch.tensor(~np.isin(data.input_ids.detach().cpu().numpy(), (0, 1, 2, 3)))
    masked_ids = (mask1 * mask2)
    data.input_ids[masked_ids] = 4
    return masked_ids


def get_teacher_student_tokenizer():
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    teacher = AutoModel.from_pretrained("allegro/herbert-base-cased")
    student = creat_student(teacher)

    return teacher, student, tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

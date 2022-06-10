import torch
import numpy as np


# czy powinniśmy zasłaniać unk token?
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
    student = AutoModel.from_pretrained("allegro/herbert-base-cased")

    # student.encoder.layer[1] = torch.nn.Identity(768)
    # student.encoder.layer[3] = torch.nn.Identity(768)
    # student.encoder.layer[5] = torch.nn.Identity(768)
    # student.encoder.layer[7] = torch.nn.Identity(768)
    # student.encoder.layer[9] = torch.nn.Identity(768)
    # student.encoder.layer[11] = torch.nn.Identity(768)

    print(f'Number of parameters: Teacher: {count_parameters(teacher)}, Student: {count_parameters(student)},'
          f'Student / Teacher ratio: {round(count_parameters(student) / count_parameters(teacher), 4)}.')

    for params in teacher.parameters():
        params.requires_grad = False

    return teacher, student, tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

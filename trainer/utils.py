import torch
import numpy as np

from models.distil_student import creat_student


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
    student = creat_student(teacher)
    
    print(f'Number of parameters: Teacher: {count_parameters(teacher)}, Student: {count_parameters(student)},'
          f'Student / Teacher ratio: {round(count_parameters(student) / count_parameters(teacher), 4)}.')

    for params in teacher.parameters():
        params.requires_grad = False

    return teacher, student, tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CudaTimer:
    def __init__(self):
        self.start = None;
        self.end = None
        self.epoch_time = 0;
        self.run_time = 0

    def start_event(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def end_event(self, epoch):
        self.end.record();
        torch.cuda.synchronize()
        time_in_sec = self.start.elapsed_time(self.end) // 1000
        self.epoch_time = time_in_sec
        self.run_time += time_in_sec

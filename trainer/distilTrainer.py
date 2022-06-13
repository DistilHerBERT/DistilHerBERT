import os
import datetime

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import create_masked_ids
from tensorboard_pytorch import TensorboardPyTorch


class DistilTrainer(object):
    def __init__(self, teacher, student, tokenizer, loaders, criterion1, criterion2, criterion3,
                 optim, accelerator=None, scheduler=None, device='cpu'):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.criterion1 = criterion1    # MLM
        self.criterion2 = criterion2    # distil
        self.criterion3 = criterion3    # cosine
        self.optim = optim
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.loaders = loaders
        self.n_logger = None  # neptune logger
        self.t_logger = None  # tensorflow logger
        self.device = device

    def run_exp(self, epoch_start, epoch_end, exp_name, save_interval, temp=0.5, random_seed=42):
        save_path = self.at_exp_start(exp_name, random_seed)
        for epoch in tqdm(range(epoch_start, epoch_end)):
            self.teacher.eval()
            self.student.train()
            self.run_epoch(epoch, save_path, save_interval, phase='train', temp=temp)
            self.student.eval()
            with torch.no_grad():
                self.run_epoch(epoch, save_path, save_interval, phase='test', temp=1.)

    def at_exp_start(self, exp_name, random_seed):
        self.manual_seed(random_seed)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(os.getcwd(), f'exps/{exp_name}/{date}')
        save_path = f'{base_path}/checkpoints'
        os.makedirs(save_path)
        self.t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', self.device)
        self.n_logger = None
        return save_path

    def run_epoch(self, epoch, save_path, save_interval, phase, temp):
        running_loss1_teacher = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss = 0.0
        loader_size = len(self.loaders[phase])
        # self.n_logger['epoch'].log(epoch)
        for i, data in enumerate(self.loaders[phase]):
            # self.n_logger['step'].log(i)
            tokenized_data = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=list(data['sentence']),
                padding='longest',
                truncation=True,
                # add_special_tokens=False,
                return_token_type_ids=False,
                return_tensors='pt'
            ).to(self.device)
            masked_ids_bool = create_masked_ids(tokenized_data).view(-1)
            masked_target_ids = tokenized_data['input_ids'].view(-1)[masked_ids_bool]

            y_pred_student = self.student(**tokenized_data)[0]
            y_pred_student = y_pred_student.view(-1, y_pred_student.size(-1))[masked_ids_bool]

            with torch.no_grad():
                y_pred_teacher = self.teacher(**tokenized_data)['last_hidden_state']
                y_pred_teacher = y_pred_teacher.view(-1, y_pred_teacher.size(-1))[masked_ids_bool]

            assert y_pred_student.shape == y_pred_teacher.shape

            loss0 = self.criterion1(y_pred_teacher, masked_target_ids)
            loss1 = self.criterion1(y_pred_student, masked_target_ids)
            loss2 = self.criterion2(F.log_softmax(y_pred_student/temp, dim=-1), F.softmax(y_pred_teacher/temp, dim=-1))
            loss3 = self.criterion3(y_pred_student,
                                    y_pred_teacher,
                                    torch.ones(masked_target_ids.shape).view(-1).to(self.device))
            # jakies ważenie losów? może związane ze schedulerem?
            loss = (loss1 + loss2 + loss3) / 3
            if 'train' in phase:
                self.optim.zero_grad()
                self.accelerator.backward(loss) # jedyne użycie acceleratora w trainerze
                # loss.backward()
                self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            # self.n_logger['train_every_step1'].log(loss1)
            # self.n_logger['train_every_step2'].log(loss2)
            # self.n_logger['train_every_step3'].log(loss3)
            # self.n_logger['train_every_step'].log(loss)

            running_loss1_teacher += loss0.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
            running_loss += loss.item()
            # loggers
            if (i + 1) % 30 == 0:
                tmp_loss0 = running_loss1_teacher / 30
                tmp_loss1 = running_loss1 / 30
                tmp_loss2 = running_loss2 / 30
                tmp_loss3 = running_loss3 / 30
                tmp_loss = running_loss / 30

                # self.n_logger[f'MLM Loss/{phase}'].log(tmp_loss1)
                # self.n_logger[f'Distil Loss/{phase}'].log(tmp_loss2)
                # self.n_logger[f'Cosine Loss/{phase}'].log(tmp_loss3)
                # self.n_logger[f'Loss/{phase}'].log(tmp_loss)

                self.t_logger.log_scalar(f'MLM Loss Teacher/{phase}', round(tmp_loss0, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'MLM Loss/{phase}', round(tmp_loss1, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Distil Loss/{phase}', round(tmp_loss2, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Cosine Loss/{phase}', round(tmp_loss3, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Loss/{phase}', round(tmp_loss, 4), i + 1 + epoch * loader_size)

                running_loss1_teacher = 0.0
                running_loss1 = 0.0
                running_loss2 = 0.0
                running_loss3 = 0.0
                running_loss = 0.0

                # if (i + 1) % save_interval == 0:
                #     self.save_student(save_path)

    def save_student(self, path):
        torch.save(self.student.state_dict(), f"{path}/student_{datetime.datetime.utcnow()}.pth")
        # self.student.save_pretrained(f"{path}/student_{datetime.datetime.utcnow()}.pth")

    def manual_seed(self, random_seed):
        import numpy as np
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.manual_seed_all(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

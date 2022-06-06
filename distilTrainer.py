import datetime

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class DistilTrainer(object):
    def __init__(self, teacher, student, criterion2, scheduler, loaders, optim, device):
        self.teacher = teacher
        self.student = student
        self.criterion1 = nn.CrossEntropyLoss()  # MLM
        self.criterion2 = criterion2  # distil
        self.criterion3 = nn.CosineEmbeddingLoss(reduction="mean")  # cosine
        self.optim = optim
        self.scheduler = scheduler
        self.loaders = loaders
        self.n_logger = None  # neptune logger
        self.t_logger = None  # tensorflow logger
        self.device = device

    def run_exp(self, t_logger, n_logger, epoch_start, epoch_end, save_path, save_interval, random_seed, writer):
        self.manual_seed(random_seed)
        self.t_logger = t_logger
        self.n_logger = n_logger
        for epoch in tqdm(range(epoch_start, epoch_end)):
            # self.model.train()
            self.run_epoch(writer, epoch, save_path, save_interval, phase='train')
            # self.model.eval()
            with torch.no_grad():
                self.run_epoch(writer, epoch, save_path, save_interval, phase='test')
            self.scheduler.step()

    def run_epoch(self, writer, epoch, save_path, save_interval, phase):
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss = 0.0
        loader_size = len(self.loaders[phase])
        self.n_logger['epoch'].log(epoch)
        for i, data in enumerate(self.loaders[phase]):
            self.n_logger['step'].log(i)
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)

            y_pred_teacher = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            y_pred_student = self.student(input_ids=input_ids, attention_mask=attention_mask)

            loss1 = self.criterion1(y_pred_student, input_ids, attention_mask)
            loss2 = self.criterion2(y_pred_student, y_pred_teacher.softmax(-1))
            loss3 = self.criterion3(y_pred_student, y_pred_teacher)
            loss = loss1 + loss2 + loss3
            if 'train' in phase:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            self.n_logger['train_every_step1'].log(loss1)
            self.n_logger['train_every_step2'].log(loss2)
            self.n_logger['train_every_step3'].log(loss3)
            self.n_logger['train_every_step'].log(loss)

            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
            running_loss += loss.item()
            # loggers
            if (i + 1) % 100 == 0:  # print every 2000 mini-batches
                tmp_loss1 = running_loss1 / 100
                tmp_loss2 = running_loss2 / 100
                tmp_loss3 = running_loss3 / 100
                tmp_loss = running_loss / 100

                self.n_logger[f'MLM Loss/{phase}'].log(tmp_loss1)
                self.n_logger[f'Distil Loss/{phase}'].log(tmp_loss2)
                self.n_logger[f'Cosine Loss/{phase}'].log(tmp_loss3)
                self.n_logger[f'Loss/{phase}'].log(tmp_loss)

                self.t_logger.log_scalar(f'MLM Loss/{phase}', round(tmp_loss1, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Distil Loss/{phase}', round(tmp_loss2, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Cosine Loss/{phase}', round(tmp_loss3, 4), i + 1 + epoch * loader_size)
                self.t_logger.log_scalar(f'Loss/{phase}', round(tmp_loss, 4), i + 1 + epoch * loader_size)

                running_loss1 = 0.0
                running_loss2 = 0.0
                running_loss3 = 0.0
                running_loss = 0.0

                if (i + 1) % save_interval == 0:
                    self.save_net(save_path)

    def save_net(self, path):
        torch.save(self.student.state_dict(), f"{path}\student_{datetime.datetime.utcnow()}.pth")
        torch.save(self.teacher.state_dict(), f"{path}\teacher_{datetime.datetime.utcnow()}.pth")
        # save_model(self.model, path)

    def manual_seed(self, random_seed):
        import numpy as np
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)

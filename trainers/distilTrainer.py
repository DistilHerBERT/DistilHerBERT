import os
import datetime

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from logger.logger import NeptuneLogger
from trainers.utils import create_masked_ids, get_masked_mask_and_att_mask
from trainers.tensorboard_pytorch import TensorboardPyTorch


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

    def run_exp(self, epoch_start, epoch_end, exp_name, config_run_epoch, temp=1.0, random_seed=42):
        save_path = self.at_exp_start(exp_name, random_seed)
        for epoch in tqdm(range(epoch_start, epoch_end), desc='run_exp'):
            self.teacher.eval()
            self.student.train()
            self.run_epoch(epoch, save_path, config_run_epoch, phase='train', temp=temp)
            self.student.eval()
            with torch.no_grad():
                self.run_epoch(epoch, save_path, config_run_epoch, phase='test', temp=1.0)

    def at_exp_start(self, exp_name, random_seed):
        self.manual_seed(random_seed)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(os.getcwd(), f'exps/{exp_name}/{date}')
        save_path = f'{base_path}/checkpoints'
        os.makedirs(save_path)
        self.t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', self.device)
        self.n_logger = NeptuneLogger(exp_name)
        return save_path

    def run_epoch(self, epoch, save_path, config_run_epoch, phase, temp):
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss = 0.0
        running_denom = 0.0
        loader_size = len(self.loaders[phase])
        self.n_logger['epoch'].log(epoch)
        progress_bar = tqdm(self.loaders[phase], desc='run_epoch', mininterval=30, leave=False, total=loader_size)
        for i, data in enumerate(progress_bar):
            tokenized_input, tokenized_output, lengths = data
            self.n_logger['step'].log(i)

            masked_mask, att_mask = get_masked_mask_and_att_mask(tokenized_output, lengths)
            masked_mask = masked_mask.view(-1)
            masked_mask, att_mask = masked_mask.to(self.device), att_mask.to(self.device)
            tokenized_output = tokenized_output.view(-1)[masked_mask]

            y_pred_student = self.student(input_ids=tokenized_input, attention_mask=att_mask)[0]
            y_pred_student = y_pred_student.view(-1, y_pred_student.size(-1))[masked_mask]
            y_pred_student = y_pred_student @ self.student.embeddings.word_embeddings.weight.T

            with torch.no_grad():
                y_pred_teacher = self.teacher(input_ids=tokenized_input, attention_mask=att_mask)['last_hidden_state']
                y_pred_teacher = y_pred_teacher.view(-1, y_pred_teacher.size(-1))[masked_mask]
                y_pred_teacher = y_pred_teacher @ self.teacher.embeddings.word_embeddings.weight.T

            assert y_pred_student.shape == y_pred_teacher.shape

            loss1 = self.criterion1(y_pred_student, tokenized_output)
            loss2 = self.criterion2(F.log_softmax(y_pred_student/temp, dim=-1), F.softmax(y_pred_teacher/temp, dim=-1))
            loss3 = self.criterion3(y_pred_student,
                                    y_pred_teacher,
                                    torch.ones(tokenized_output.shape).to(self.device))

            # jakies ważenie losów? może związane ze schedulerem?
            loss = (loss1 + loss2 + loss3) / 3

            self.n_logger['train_every_step1_mlm_loss'].log(loss1.item())
            self.n_logger['train_every_step2_distill_loss'].log(loss2.item())
            self.n_logger['train_every_step3_cosine_loss'].log(loss3.item())
            self.n_logger['train_every_step'].log(loss.item())

            loss /= config_run_epoch.grad_accum_steps
            if 'train' in phase:
                # loss.backward()
                self.accelerator.backward(loss) # jedyne użycie acceleratora w trainerze
                if (i + 1) % config_run_epoch.grad_accum_steps == 0:
                    self.accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.student.parameters()), 1.)
                    self.optim.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.zero_grad()
            loss *= config_run_epoch.grad_accum_steps

            denom = tokenized_output.size(0)
            running_loss1 += loss1.item() * denom
            running_loss2 += loss2.item() * denom
            running_loss3 += loss3.item() * denom
            running_loss += loss.item() * denom
            running_denom += denom
            # loggers
            if (i + 1) % config_run_epoch.running_step == 0:
                tmp_loss1 = running_loss1 / (config_run_epoch.running_step * running_denom)
                tmp_loss2 = running_loss2 / (config_run_epoch.running_step * running_denom)
                tmp_loss3 = running_loss3 / (config_run_epoch.running_step * running_denom)
                tmp_loss = running_loss / (config_run_epoch.running_step * running_denom)

                progress_bar.set_postfix({'mlm': tmp_loss1, 'distil': tmp_loss2, 'cosine': tmp_loss3, 'loss': tmp_loss})

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
                running_denom = 0.0

                # if (i + 1) % config_run_epoch.save_interval == 0:
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

import os
import datetime

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from logger.logger import NeptuneLogger
from trainers.utils import create_masked_ids, get_masked_mask_and_att_mask
from trainers.tensorboard_pytorch import TensorboardPyTorch


class VanillaTrainer(object):
    def __init__(self, model, tokenizer, loaders, criterion,
                 optim, accelerator=None, scheduler=None, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optim = optim
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.loaders = loaders
        self.n_logger = None  # neptune logger
        self.t_logger = None  # tensorflow logger
        self.device = device

    def run_exp(self, epoch_start, epoch_end, exp_name, config_run_epoch, random_seed=42):
        save_path = self.at_exp_start(exp_name, random_seed)
        for epoch in tqdm(range(epoch_start, epoch_end), desc='run_exp'):
            self.model.train()
            self.run_epoch(epoch, save_path, config_run_epoch, phase='train')
            self.model.eval()
            with torch.no_grad():
                self.run_epoch(epoch, save_path, config_run_epoch, phase='test')

    def at_exp_start(self, exp_name, random_seed):
        self.manual_seed(random_seed)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(os.getcwd(), f'exps/{exp_name}/{date}')
        save_path = f'{base_path}/checkpoints'
        os.makedirs(save_path)
        self.t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', self.device)
        self.n_logger = NeptuneLogger(exp_name)
        return save_path

    def run_epoch(self, epoch, save_path, config_run_epoch, phase):
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

            y_pred = self.model(input_ids=tokenized_input, attention_mask=att_mask)[0]
            y_pred = y_pred.view(-1, y_pred.size(-1))[masked_mask]
            y_pred = y_pred @ self.model.embeddings.word_embeddings.weight.T

            loss = self.criterion(y_pred, tokenized_output)

            self.n_logger['train_every_step_mlm_loss'].log(loss.item())

            loss /= config_run_epoch.grad_accum_steps
            if 'train' in phase:
                # loss.backward()
                self.accelerator.backward(loss) # jedyne użycie acceleratora w trainerze, razem z clip_grad..
                if (i + 1) % config_run_epoch.grad_accum_steps == 0:
                    self.accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.)
                    self.optim.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.zero_grad()
            loss *= config_run_epoch.grad_accum_steps

            denom = tokenized_output.size(0)
            running_loss += loss.item() * denom
            running_denom += denom
            # loggers
            if (i + 1) % config_run_epoch.running_step == 0:
                tmp_loss = running_loss / (config_run_epoch.running_step * running_denom)

                progress_bar.set_postfix({'mlm_loss': tmp_loss})

                self.n_logger[f'MLM Loss/{phase}'].log(tmp_loss)
                self.t_logger.log_scalar(f'MLM Loss/{phase}', round(tmp_loss, 4), i + 1 + epoch * loader_size)

                running_loss = 0.0
                running_denom = 0.0

                # if (i + 1) % config_run_epoch.save_interval == 0:
                #     self.save_student(save_path)

    def save_student(self, path):
        torch.save(self.model.state_dict(), f"{path}/model_{datetime.datetime.utcnow()}.pth")

    def manual_seed(self, random_seed):
        import numpy as np
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.manual_seed_all(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

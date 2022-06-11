import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardPyTorch:
    def __init__(self, log_name, device='cpu'):
        self.writer = SummaryWriter(log_dir=log_name, flush_secs=60)
        self.device = device
        self.tensors = {}

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

#     def update_tensors(self, name, value):
#         if name in self.tensors:
#             self.tensors[name] = torch.cat((self.tensors[name], value))
#         else:
#             self.tensors[name] = value

#     def release_tensors(self):
#         self.tensors = {}

    def log_graph(self, model, inp):
        self.writer.add_graph(model, inp)

    def log_histogram(self, tag, tensor, global_step=0):
        self.writer.add_histogram(tag, tensor, global_step)
        self.flush()

    def log_scalar(self, tag, scalar, global_step):
        self.writer.add_scalar(tag, scalar, global_step)
        self.flush()
        
    def log_scalars(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        self.flush()

    # def log_embeddings(self, model, dataset, epoch):
    #     self.writer.add_embedding(
    #         model.forward(get_val_data(dataset, 'data').to(self.device)),
    #         metadata=get_val_data(dataset, 'target'),
    #         label_img=get_val_data(dataset, 'data'),
    #         global_step=epoch)
    #     self.flush()

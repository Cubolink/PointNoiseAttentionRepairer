import os
import json
import torch

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossHistory:
    def __init__(self, metrics: list[str], filename:str):
        # TODO: support multiple losses

        self.epoch_list = []
        self.iteration_list = []
        self.loss_dict = {k: [] for k in metrics}
        self.filename = filename

    def load(self, filename):
        if os.path.exists(filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
            self.epoch_list = data['epoch']
            self.iteration_list = data['it']
            self.loss_dict = data['loss']

    def update(self, epoch, it, losses):
        self.epoch_list.append(epoch)
        self.iteration_list.append(it)
        for key, val in losses.items():
            self.loss_dict[key].append(val)

    def save(self):
        data = {
            'epoch': self.epoch_list,
            'it': self.iteration_list,
            'loss': self.loss_dict,
        }
        with open(self.filename, 'w') as f:
            json.dump(data, f)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)









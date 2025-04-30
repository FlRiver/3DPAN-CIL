import random
import numpy as np
import torch
import os
import logging
import torch.distributed as dist
import shutil
import torch.nn as nn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def save_model(net, epoch, T, path, acc, acc_is_best, loss_is_best, **kwargs):
    state = {
        'net': net.state_dict(),
        'task': T,
        'epoch': epoch,
        'acc': acc
    }
    for key, value in kwargs.items():
        state[key] = value
    
    path = os.path.join(path, 'pths')
    os.makedirs(path, exist_ok=True)
    
    filepath = os.path.join(path, f'checkpoint_{T}_last.pth')
    torch.save(state, filepath)
    if acc_is_best:
        shutil.copyfile(filepath, os.path.join(path, f'checkpoint_{T}_acc_best.pth'))
    if loss_is_best:
        shutil.copyfile(filepath, os.path.join(path, f'checkpoint_{T}_loss_best.pth'))


def save_args(args):
    os.makedirs(args.experiment_path, exist_ok=True)
    file = open(os.path.join(args.experiment_path, 'log_args.txt'), "w")
    for k, v in vars(args).items():
        file.write(f"{k}:\t {v}\n")
    file.close()


def write_lists_to_txt(Acc_list, AFR_list, file_path):
    with open(file_path, 'w') as file:
        file.write(f"Task {len(AFR_list)}:\n")
        # 写入 Acc_list
        file.write("Acc:\n")
        for acc_row in Acc_list:
            for value in acc_row:
                file.write(f"{value}\t")
            file.write("\n")
        file.write("\n")
        # 写入 AFR_list
        file.write("AFR:\n")
        for value in AFR_list:
            file.write(f"{value}\t")
        file.write("\n")


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, resume=False): 
        self.file = None
        self.resume = resume
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (confidence * nll_loss + self.smoothing * smooth_loss).mean()
        return loss
"""

Training Logger

"""
import torch


class Logger:

    def __init__(self):
        pass


def save_checkpoints(model, optimizer, epoch, iteration, path):

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration
    }

    torch.save(state_dict, path)


def load_checkpoints(path):
    state_dict = torch.load(path)

    return state_dict['model'], state_dict['optimizer'], state_dict['epoch'], state_dict['iteration']


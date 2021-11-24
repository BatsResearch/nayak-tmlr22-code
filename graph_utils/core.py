import os
import random

import torch
import numpy as np


def create_dirs(path):
    """create directories if path doesn't exist

    Arguments:
        path {str} -- path of the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, save_path):
    """The function is used to save the model

    Arguments:
        model {nn.Model} -- the model
        save_path {str} -- model save path
    """
    # TODO: test this module
    torch.save(model.state_dict(), save_path)

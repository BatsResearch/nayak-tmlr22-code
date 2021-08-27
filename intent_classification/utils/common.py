import os
import json
import random
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp


def set_seed(seed):
    """function sets the seed value

    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_save_path(dir_path, options):
    """Function returns the save path, given
    the options and the base directory path.

    Args:
        dir_path (str): base directory path.
        options (dict): options used for training.

    Returns:
        str: save path
    """
    save_path = options["label_encoder_type"]

    save_path += "_seed_" + str(options["seed"])

    save_path += ".pt"

    save_path = os.path.join(dir_path, save_path)

    return save_path


def create_dirs(dir_path):
    """Function to create directory if not present.

    Args:
        dir_path (str): path of the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def init_device(gpu=0):
    """Function is used to get the cuda device
    if available.

    Args:
        gpu (int, optional): cuda device id. Defaults to 0.

    Returns:
        tuple: torch.device and cuda_device id
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
        cuda_device = gpu
    else:
        device = torch.device("cpu")
        cuda_device = -1
    return device, cuda_device


def normt_spm(mx, method="in"):
    if method == "in":
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == "sym":
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col))
    ).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

import json
import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


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


def fine_labels(dataset, level):
    """Function to split the labels into two types - coarse-grained
    and fine-grained types. The coarase-grained types are called
    train_labels and fine-grained types are called test_labels.

    Args:
        dataset (list): list of lines
        level (int): number of levels for the fine-grained type.

    Returns:
        tuple: train_labels and test_labels
    """
    labels = []
    train_labels = []
    test_labels = []
    for line in dataset:
        example = json.loads(line)
        labels += example["labels"]

    #
    labels = list(set(labels))
    for label in labels:
        if len(label.split("/")) == level + 1:
            test_labels.append(label)
        else:
            train_labels.append(label)

    return train_labels, test_labels


def flatten_dataset(lines):
    """Function to flatten the list of lines.
    This is to make the dataset easier to load.

    Args:
        lines (list): list of dataset lines

    Returns:
        list: list of lines with flattened dataset lines
    """
    dataset = []
    for line in lines:
        _temp = []
        example = json.loads(line)
        for mention in example["mentions"]:
            instance = {
                "tokens": example["tokens"],
                "senid": example["senid"],
                "fileid": example["fileid"],
                "labels": mention["labels"],
                "start": mention["start"],
                "end": mention["end"],
            }
            _temp.append(json.dumps(instance))

        dataset += _temp

    return dataset


def remove_labels(dataset, train_labels):
    """Function to include only train_labels
    in the dataset.

    Args:
        dataset (list): list of dataset lines
        train_labels (list): list of train labels

    Returns:
        list: list of dataset lines with only train labels
    """
    clean_dataset = []
    for line in dataset:
        example = json.loads(line)
        _labels = [
            _label for _label in example["labels"] if _label in train_labels
        ]
        if _labels:
            example["labels"] = _labels
            clean_dataset.append(json.dumps(example))

    return clean_dataset


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

import numpy as np
import scipy.sparse as sp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils.common import normt_spm, spm_to_tensor


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class SGCN(nn.Module):
    def __init__(self, n, edges, features, label_idx, device):
        super().__init__()

        edges = np.array(edges)
        adj = sp.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(n, n),
            dtype="float32",
        )
        adj = normt_spm(adj, method="in")
        adj = spm_to_tensor(adj)
        self.adj = adj.to(device)
        self.feat = features.to(device)

        i = 0
        layers = []

        conv = GraphConv(300, 64, dropout=True)
        self.add_module("conv{}".format(i), conv)
        layers.append(conv)

        i += 1
        conv = GraphConv(64, 64, dropout=True)
        self.add_module("conv{}".format(i), conv)
        layers.append(conv)

        self.output_dim = 64
        self.label_idx = label_idx

        self.layers = layers

    def forward(self, train=False):
        x = copy.deepcopy(self.feat)
        for conv in self.layers:
            x = conv(x, self.adj)
            x = F.normalize(x)

        label_preds = x[self.label_idx]

        return label_preds

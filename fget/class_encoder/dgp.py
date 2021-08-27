"""Obtained code from https://github.com/cyvius96/DGP
"""


import copy
import numpy as np
import scipy.sparse as sp

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

    def forward(self, inputs, adj_set, att):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        support = torch.mm(inputs, self.w) + self.b
        outputs = None
        for i, adj in enumerate(adj_set):
            y = torch.mm(adj, support) * att[i]
            if outputs is None:
                outputs = y
            else:
                outputs = outputs + y

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class DGP(nn.Module):
    def __init__(self, n, edges_set, features, device):
        super().__init__()

        self.n = n
        self.d = len(edges_set)
        self.feat = features

        self.a_adj_set = []
        self.r_adj_set = []

        for edges in edges_set:
            edges = np.array(edges)
            adj = sp.coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                shape=(n, n),
                dtype="float32",
            )
            a_adj = spm_to_tensor(normt_spm(adj, method="in")).to(device)
            r_adj = spm_to_tensor(normt_spm(adj.transpose(), method="in")).to(
                device
            )
            self.a_adj_set.append(a_adj)
            self.r_adj_set.append(r_adj)

        self.a_att = nn.Parameter(torch.ones(self.d).to(device))
        self.r_att = nn.Parameter(torch.ones(self.d).to(device))

        i = 0
        layers = []

        conv = GraphConv(300, 128, dropout=True)
        self.add_module("conv{}".format(i), conv)
        layers.append(conv)

        i += 1
        conv = GraphConv(128, 128, dropout=True)
        self.add_module("conv{}".format(i), conv)
        layers.append(conv)

        self.output_dim = 128

        self.layers = layers

    def forward(self, label_idx):
        graph_side = True
        x = copy.deepcopy(self.feat)
        for conv in self.layers:
            if graph_side:
                adj_set = self.a_adj_set
                att = self.a_att
            else:
                adj_set = self.r_adj_set
                att = self.r_att
            att = F.softmax(att, dim=0)
            x = conv(x, adj_set, att)
            x = F.normalize(x)
            graph_side = not graph_side

        label_preds = x[label_idx, :]

        return label_preds

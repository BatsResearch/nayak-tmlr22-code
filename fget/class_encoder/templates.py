## config templates
from zsl_kg.common.graph import NeighSampler

import torch.nn as nn

## Configs for AutoGNN
gcn = {
    "input_dim": 300,
    "output_dim": 128,
    "type": "gcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 128,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

gat = {
    "input_dim": 300,
    "output_dim": 128,
    "type": "gat",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 128,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

rgcn = {
    "input_dim": 300,
    "output_dim": 128,
    "type": "rgcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "add_weight": True,
            "sampler": NeighSampler(100, mode="topk"),
            "num_basis": 1,
            "num_rel": 50 + 1,
            "self_rel_id": 50,
        },
        {
            "input_dim": 128,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "add_weight": True,
            "sampler": NeighSampler(50, mode="topk"),
            "num_basis": 1,
            "num_rel": 50 + 1,
            "self_rel_id": 50,
        },
    ],
}

lstm = {
    "input_dim": 300,
    "output_dim": 128,
    "type": "lstm",
    "gnn": [
        {
            "input_dim": 300,
            "lstm_dim": 300,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 128,
            "lstm_dim": 128,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

trgcn = {
    "input_dim": 300,
    "output_dim": 128,
    "type": "trgcn",
    "gnn": [
        {
            "input_dim": 300,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(100, mode="topk"),
        },
        {
            "input_dim": 128,
            "output_dim": 128,
            "activation": nn.ReLU(),
            "normalize": True,
            "sampler": NeighSampler(50, mode="topk"),
        },
    ],
}

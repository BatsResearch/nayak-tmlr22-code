import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from class_encoder.dgp import DGP
from class_encoder.gcnz import GCNZ
from class_encoder.sgcn import SGCN
from class_encoder.templates import gat, gcn, lstm, rgcn, trgcn
from zsl_kg.class_encoders.auto_gnn import AutoGNN

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


GNN_CONFIGS = {
    "gcn": gcn,
    "gat": gat,
    "rgcn": rgcn,
    "trgcn": trgcn,
    "lstm": lstm,
}


def get_label_encoder(label_encoder_type, options=None):
    if label_encoder_type == "gcnz":
        return gcnz_encoder(options)
    elif label_encoder_type == "dgp":
        return dgp_encoder(options)
    elif label_encoder_type == "sgcn":
        return sgcn_encoder(options)
    else:
        return AutoGNN(GNN_CONFIGS[label_encoder_type])


def dgp_encoder(options):

    wn_mapping = pd.read_csv(
        os.path.join(DIR_PATH, "../misc_data/snips_mapping.csv")
    )

    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/dense_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)

    edges_set = graph["edges_set"]
    print("edges_set", [len(l) for l in edges_set])

    # this is the K value; this indicates the depth of ancestors and
    # descendants.
    # assuming this is right;
    lim = 4
    for i in range(lim + 1, len(edges_set)):
        edges_set[lim].extend(edges_set[i])
    edges_set = edges_set[: lim + 1]
    print("edges_set", [len(l) for l in edges_set])

    word_vectors = torch.tensor(graph["vectors"])
    word_vectors = F.normalize(word_vectors)

    wnid_to_idx = dict([(wnid, idx) for idx, wnid in enumerate(wnids)])
    label_idx = [wnid_to_idx[wn_mapping["wnid"][i]] for i in range(7)]

    label_encoder = DGP(
        n, edges_set, word_vectors, label_idx, options["device"]
    )

    return label_encoder


def sgcn_encoder(options):

    wn_mapping = pd.read_csv(
        os.path.join(DIR_PATH, "../misc_data/snips_mapping.csv")
    )

    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    wnid_to_idx = dict([(wnid, idx) for idx, wnid in enumerate(wnids)])
    label_idx = [wnid_to_idx[wn_mapping["wnid"][i]] for i in range(7)]

    word_vectors = torch.tensor(graph["vectors"])
    word_vectors = F.normalize(word_vectors)

    label_encoder = SGCN(n, edges, word_vectors, label_idx, options["device"])
    return label_encoder


def gcnz_encoder(options):

    wn_mapping = pd.read_csv(
        os.path.join(DIR_PATH, "../misc_data/snips_mapping.csv")
    )

    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    # wn_mapping = pd.read_csv(os.path.join(DIR_PATH, 'mapping/'+ dataset +'/wnid_mapping.csv'))
    wnid_to_idx = dict([(wnid, idx) for idx, wnid in enumerate(wnids)])

    label_idx = [wnid_to_idx[wn_mapping["wnid"][i]] for i in range(7)]
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph["vectors"])
    word_vectors = F.normalize(word_vectors)

    label_encoder = GCNZ(n, edges, word_vectors, label_idx, options["device"])

    return label_encoder

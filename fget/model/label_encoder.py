import json
import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util as nn_util
from allennlp.nn.util import get_text_field_mask, masked_mean
from class_encoder.avg_label import AvgLabel
from class_encoder.description import DescEncoder
from class_encoder.dgp import DGP
from class_encoder.gcnz import GCNZ
from class_encoder.sgcn import SGCN
from class_encoder.templates import gat, gcn, lstm, rgcn, trgcn
from zsl_kg.class_encoders.auto_gnn import AutoGNN
from zsl_kg.example_encoders.text_encoder import TextEncoder
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

GNN_CONFIGS = {
    "gcn": gcn,
    "gat": gat,
    "rgcn": rgcn,
    "trgcn": trgcn,
    "lstm": lstm,
}


GLOVE_PATH = os.path.join(DIR_PATH, "../data/glove.840B.300d.txt")


def get_label_encoder(options):

    label_encoder_type = options["label_encoder_type"]

    if label_encoder_type == "otyper":
        return avg_label_class_encoder(options)

    elif label_encoder_type == "dzet":
        return desc_class_encoder(options)

    elif label_encoder_type == "sgcn":
        return sgcn_class_encoder(options)

    elif label_encoder_type == "gcnz":
        return gcnz_class_encoder(options)

    elif label_encoder_type == "dgp":
        return dgp_class_encoder(options)

    else:
        return graph_class_encoder(options)


def avg_label_class_encoder(options):
    all_tokens = []
    list_of_token_list = []

    dataset_path = options["dataset_path"]

    train_df = pd.read_csv(os.path.join(dataset_path, "train_labels.csv"))
    train_labels = train_df["LABELS"].to_list()

    test_df = pd.read_csv(os.path.join(dataset_path, "test_labels.csv"))
    test_labels = test_df["LABELS"].to_list()
    all_labels = train_labels + test_labels

    for i, label in enumerate(all_labels):
        label_words = label.lstrip("/").split("/")
        # the individual labels also have "_" between them
        label_words = [
            word
            for partial_label in label_words
            for word in partial_label.split("_")
        ]
        all_tokens.extend(label_words)
        list_of_token_list.append(label_words)

    label_counter = Counter(all_tokens)
    label_vocab = Vocabulary({"tokens": label_counter})

    token_embedding = Embedding.from_params(
        vocab=label_vocab,
        params=Params(
            {
                "pretrained_file": GLOVE_PATH,
                "embedding_dim": 300,
                "trainable": False,
            }
        ),
    )

    token_to_idx = label_vocab.get_token_to_index_vocabulary("tokens")

    padded_idx = convert_token_to_idx(list_of_token_list, token_to_idx)
    mask = get_text_field_mask({"tokens": padded_idx})

    padded_embs = token_embedding(padded_idx)

    avg_label_tensor = masked_mean(padded_embs, mask.unsqueeze(-1), dim=1)
    avg_label_emb = nn.Embedding.from_pretrained(avg_label_tensor, freeze=True)

    avg_label_encoder = AvgLabel(avg_label_emb)

    return avg_label_encoder


def desc_class_encoder(options):
    dataset = options["dataset"]
    dataset_path = options["dataset_path"]

    with open(
        os.path.join(DIR_PATH, "../misc_data/" + dataset + "/desc.json")
    ) as fp:
        desc_data = json.load(fp)

    # load the vocab
    words = []
    for desc_tokens in desc_data.values():
        words += desc_tokens

    desc_vocab = Vocabulary({"tokens": Counter(words)})

    desc_emb = Embedding.from_params(
        vocab=desc_vocab,
        params=Params(
            {
                "pretrained_file": GLOVE_PATH,
                "embedding_dim": 300,
                "trainable": False,
            }
        ),
    )

    word_embeddings = BasicTextFieldEmbedder({"tokens": desc_emb})
    #

    # get the text encoder
    text_encoder = TextEncoder(
        word_embeddings, input_dim=300, hidden_dim=100, attn_dim=100
    )

    train_df = pd.read_csv(os.path.join(dataset_path, "train_labels.csv"))
    train_labels = train_df["LABELS"].to_list()

    test_df = pd.read_csv(os.path.join(dataset_path, "test_labels.csv"))
    test_labels = test_df["LABELS"].to_list()
    all_labels = train_labels + test_labels

    list_of_token_list = []
    for label in all_labels:
        list_of_token_list.append(desc_data[label])

    token_to_idx = desc_vocab.get_token_to_index_vocabulary()
    description_tensor = convert_token_to_idx(
        list_of_token_list, token_to_idx
    ).to(options["device"])

    description_dict = {"tokens": description_tensor}

    nn_util.move_to_device(description_dict, options["cuda_device"])

    desc_label_encoder = DescEncoder(text_encoder, description_dict)

    return desc_label_encoder


def sgcn_class_encoder(options):
    device = options["device"]
    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph["vectors"])
    word_vectors = F.normalize(word_vectors)

    return SGCN(n, edges, word_vectors, device)


def gcnz_class_encoder(options):
    device = options["device"]
    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph["vectors"])
    word_vectors = F.normalize(word_vectors)

    return GCNZ(n, edges, word_vectors, device)


def dgp_class_encoder(options):
    device = options["device"]
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
    word_vectors = F.normalize(word_vectors).to(device)

    return DGP(n, edges_set, word_vectors, device)


def graph_class_encoder(options):
    label_encoder_type = options["label_encoder_type"]

    if options["dataset"] == "bbn":
        GNN_CONFIGS["trgcn"]["gnn"][0]["fh"] = 250

    return AutoGNN(GNN_CONFIGS[label_encoder_type])


def get_graph(graph_path):

    train_kg = ConceptNetKG.load_from_disk(
        os.path.join(graph_path, "train_graph")
    )
    test_kg = ConceptNetKG.load_from_disk(
        os.path.join(graph_path, "test_graph")
    )

    return [train_kg, test_kg]


def convert_token_to_idx(list_of_token_list, token_to_idx):
    """The code convert list of string tokens to its ids and returns
    the tensor.
    Arguments:
        list_of_token_list {list} -- list of list containing token strings
        token_to_idx {dict} -- token to id mapping from description vocab
    Returns:
        torch.tensor -- the tensor with the ids
    """
    # pad the tokens as well
    max_length = max([len(tokens) for tokens in list_of_token_list])

    token_idx_list = []

    for tokens in list_of_token_list:
        tokens_idx = [token_to_idx[token] for token in tokens]
        tokens_idx += [token_to_idx["@@PADDING@@"]] * (
            max_length - len(tokens_idx)
        )
        token_idx_list.append(tokens_idx)

    token_tensor = torch.tensor(token_idx_list)

    return token_tensor


def get_label_idx(options, test_kg=None):
    label_path = os.path.join(options["dataset_path"], "train_labels.csv")
    train_labels = pd.read_csv(label_path)
    train_len = len(train_labels["LABELS"].to_list())

    test_label_path = os.path.join(options["dataset_path"], "test_labels.csv")
    test_labels = pd.read_csv(test_label_path)

    all_labels = (
        train_labels["LABELS"].to_list() + test_labels["LABELS"].to_list()
    )
    if options["label_encoder_type"] in [
        "gcn",
        "gat",
        "lstm",
        "rgcn",
        "trgcn",
    ]:
        mapping_file = os.path.join(
            DIR_PATH,
            "../misc_data/" + options["dataset"] + "/id_to_concept.json",
        )
        mapping = json.load(open(mapping_file))
        train_names = [mapping[str(idx)] for idx in range(train_len)]
        train_idx = test_kg.get_node_ids(train_names)
        test_names = [mapping[str(idx)] for idx in range(len(mapping))]
        test_idx = test_kg.get_node_ids(test_names)

        seen_idx = [idx for idx in range(train_len)]
        unseen_idx = [
            idx for idx in range(len(test_names)) if idx not in seen_idx
        ]

    elif options["label_encoder_type"] in ["sgcn", "gcnz", "dgp"]:
        graph = json.load(
            open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
        )
        wnids = graph["wnids"]

        wnid_to_idx = dict([(wnid, idx) for idx, wnid in enumerate(wnids)])
        mapping_file = os.path.join(
            DIR_PATH, "../misc_data/" + options["dataset"] + "/wordnet.json"
        )
        mapping = json.load(open(mapping_file))

        train_idx = [
            wnid_to_idx[mapping[i]] for i in train_labels["LABELS"].to_list()
        ]
        all_labels = (
            train_labels["LABELS"].to_list() + test_labels["LABELS"].to_list()
        )
        test_idx = [wnid_to_idx[mapping[i]] for i in all_labels]

        seen_idx = [idx for idx in range(train_len)]
        unseen_idx = [
            idx for idx in range(len(mapping)) if idx not in seen_idx
        ]
    else:
        seen_idx = [idx for idx in range(train_len)]
        unseen_idx = [
            idx for idx in range(len(all_labels)) if idx not in seen_idx
        ]
        train_idx = seen_idx
        test_idx = seen_idx + unseen_idx

    return seen_idx, unseen_idx, train_idx, test_idx

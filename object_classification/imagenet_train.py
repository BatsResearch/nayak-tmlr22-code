# /users/nnayak2/data/nnayak2/zsl_image_classification/2021_pred_imagenet_save/only_imagenet
import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from tqdm import tqdm
from zsl_kg.class_encoders.auto_gnn import AutoGNN
from zsl_kg.common.graph import NeighSampler
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG
from zsl_kg.knowledge_graph.wordnet import WordNetKG

from models.templates import gat, gcn, lstm, rgcn, trgcn
from train import train_gnn_model
from utils.common import l2_loss, mask_l2_loss, set_seed

GNN_CONFIGS = {
    "gcn": gcn,
    "gat": gat,
    "rgcn": rgcn,
    "trgcn": trgcn,
    "lstm": lstm,
}


manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def pred_imagenet(model, options):
    wordnet_kg = WordNetKG.load_from_disk(options["wordnet_graph"])
    wordnet_kg.features = F.normalize(wordnet_kg.features)
    wordnet_kg.to(options["device"])
    n = len(wordnet_kg.nodes)
    imagenet_idx = [i for i in range(n)]

    model.conv.enc.aggregator.sampler.disable()
    model.conv.enc.aggregator.features.features.enc.aggregator.sampler.disable()

    model.eval()
    wnid_vectors = torch.zeros((n, 2049), device=options["device"])
    with torch.no_grad():
        for i in range(0, n, 1):
            max_idx = min(n, i + 1)
            _idx = list(range(i, max_idx))
            imagenet_idx = torch.tensor(_idx, device=options["device"])
            pred_vectors = model(imagenet_idx, wordnet_kg)
            wnid_vectors[_idx, :] = pred_vectors

    return wordnet_kg.nodes, wnid_vectors


def get_label_encoder(label_encoder_type, options):
    config = GNN_CONFIGS[label_encoder_type]
    config["gnn"][0]["sampler"] = NeighSampler(200, mode="topk")
    config["gnn"][1]["sampler"] = NeighSampler(100, mode="topk")
    model = AutoGNN(config)
    save_path = os.path.join(
        DIR_PATH,
        f'imagenet_save/{label_encoder_type}_seed_{options["seed"]}',
    )
    return model, save_path


def get_fc(resnet_50_flag=False):
    if resnet_50_flag:
        resnet = resnet50(pretrained=True)
    else:
        resnet = resnet101(pretrained=True)

    with torch.no_grad():
        b = resnet.fc.bias.detach()
        w = resnet.fc.weight.detach()
        fc_vectors = torch.cat((w, b.unsqueeze(-1)), dim=1)
    return F.normalize(fc_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_encoder_type", help="label encoder")
    parser.add_argument("--trainval", default="10,0")
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resnet-50", action="store_true")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("device : ", device)

    set_seed(int(args.seed))

    save_path = os.path.join(DIR_PATH, "imagenet_save")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ilsvrc_graph_path = os.path.join(DIR_PATH, "data/subgraphs/ilsvrc_graph")
    wordnet_graph = os.path.join(DIR_PATH, "data/subgraphs/wordnet_graph")

    options = {
        "label_encoder_type": args.label_encoder_type,
        "trainval": args.trainval,
        "num_epochs": args.max_epoch,
        "device": device,
        "seed": args.seed,
        "ilsvrc_graph_path": ilsvrc_graph_path,
        "resnet_50_flag": args.resnet_50,
        "batch_size": args.batch_size,
        "wordnet_graph": wordnet_graph,
    }

    model, save_path = get_label_encoder(
        options["label_encoder_type"], options
    )
    model = model.to(device)

    fc_vectors = get_fc(args.resnet_50)
    fc_vectors = fc_vectors.to(device)

    model, tr_log = train_gnn_model(model, fc_vectors, device, options)

    torch.save(model.state_dict(), save_path + ".pt")
    torch.save(tr_log, save_path + "_loss.json")

    wnids, output_vectors = pred_imagenet(model, options)
    torch.save({"wnids": wnids, "pred": output_vectors}, save_path + ".pred")

    print("saving complete!", save_path)

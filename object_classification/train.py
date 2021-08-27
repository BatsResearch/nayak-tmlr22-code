import argparse
import copy
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG

from models.label_encoder import get_label_encoder
from utils.common import l2_loss, mask_l2_loss, pick_vectors, set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def predict(model, graph_path, dataset):
    print("generating graph embeddings for {}".format(dataset))
    kg = ConceptNetKG.load_from_disk(graph_path)
    concepts_path = os.path.join(DIR_PATH, f"datasets/{dataset}_concepts.txt")
    concepts = [line.strip() for line in open(concepts_path)]
    concept_idx = torch.tensor(kg.get_node_ids(concepts)).to(options["device"])
    kg.to(options["device"])

    model.eval()
    pred_vectors = model(concept_idx, kg)

    return pred_vectors


def get_apy_preds(pred_obj):
    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    with open(os.path.join(DIR_PATH, "materials/apy_wnid.json")) as fp:
        apy_wnid = json.load(fp)

    train_wnids = apy_wnid["train"]
    test_wnids = apy_wnid["test"]

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )

    return pred_vectors


def get_awa_preds(pred_obj):
    awa2_split = json.load(
        open(os.path.join(DIR_PATH, "materials/awa2-split.json"), "r")
    )
    train_wnids = awa2_split["train"]
    test_wnids = awa2_split["test"]

    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )
    return pred_vectors


def train_baseline_model(model, fc_vectors, device):
    model.to(device)

    graph = json.load(
        open(os.path.join(DIR_PATH, "data/dense_graph.json"), "r")
    )
    wnids = graph["wnids"]
    word_vectors = torch.tensor(graph["vectors"]).to(device)
    word_vectors = F.normalize(word_vectors)

    print("word vectors:", word_vectors.shape)
    print("fc vectors:", fc_vectors.shape)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )

    v_train, v_val = 0.95, 0.05
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print("num train: {}, num val: {}".format(n_train, n_trainval - n_train))

    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    trlog = {}
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["min_loss"] = 0
    best_model = None

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        output_vectors = model(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        output_vectors = model(word_vectors)
        train_loss = mask_l2_loss(
            output_vectors, fc_vectors, tlist[:n_train]
        ).item()
        if v_val > 0:
            val_loss = mask_l2_loss(
                output_vectors, fc_vectors, tlist[n_train:]
            ).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print(
            "epoch {}, train_loss={:.4f}, val_loss={:.4f}".format(
                epoch, train_loss, val_loss
            )
        )

        pred_obj = {"wnids": wnids, "pred": output_vectors}

        if trlog["val_loss"]:
            min_val_loss = min(trlog["val_loss"])
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(model.state_dict())
        else:
            best_model = copy.deepcopy(model.state_dict())

        trlog["train_loss"].append(train_loss)
        trlog["val_loss"].append(val_loss)

    model.load_state_dict(best_model)
    return model, pred_obj, trlog


def train_gnn_model(model, fc_vectors, device, options):
    kg = ConceptNetKG.load_from_disk(options["ilsvrc_graph_path"])
    kg.to(options["device"])

    concepts_path = os.path.join(DIR_PATH, "datasets/ilsvrc_concepts.txt")
    concepts = [line.strip() for line in open(concepts_path)]
    ilsvrc_idx = torch.tensor(kg.get_node_ids(concepts)).to(options["device"])

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )

    v_train, v_val = map(float, options["trainval"].split(","))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print("num train: {}, num val: {}".format(n_train, n_trainval - n_train))

    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    trlog = {}
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["min_loss"] = 0
    num_w = fc_vectors.shape[0]
    best_model = None

    for epoch in range(1, options["num_epochs"] + 1):
        model.train()
        for i, start in enumerate(range(0, n_train, 100)):
            end = min(start + 100, n_train)
            indices = tlist[start:end]
            output_vectors = model(ilsvrc_idx[indices], kg)
            loss = l2_loss(output_vectors, fc_vectors[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        output_vectors = torch.empty(num_w, 2049, device=device)
        with torch.no_grad():
            for start in range(0, num_w, 100):
                end = min(start + 100, num_w)
                output_vectors[start:end] = model(ilsvrc_idx[start:end], kg)

        train_loss = mask_l2_loss(
            output_vectors, fc_vectors, tlist[:n_train]
        ).item()
        if v_val > 0:
            val_loss = mask_l2_loss(
                output_vectors, fc_vectors, tlist[n_train:]
            ).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss

        print(
            "epoch {}, train_loss={:.4f}, val_loss={:.4f}".format(
                epoch, train_loss, val_loss
            )
        )

        # check if I need to save the model
        if trlog["val_loss"]:
            min_val_loss = min(trlog["val_loss"])
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(model.state_dict())
        else:
            best_model = copy.deepcopy(model.state_dict())

        trlog["train_loss"].append(train_loss)
        trlog["val_loss"].append(val_loss)

    model.load_state_dict(best_model)
    return model, trlog


def get_fc():
    resnet = resnet101(pretrained=True)
    with torch.no_grad():
        b = resnet.fc.bias.detach()
        w = resnet.fc.weight.detach()
        fc_vectors = torch.cat((w, b.unsqueeze(-1)), dim=1)
    return F.normalize(fc_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_encoder_type", help="label encoder")
    parser.add_argument("--trainval", default="0.95,0.05")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("device : ", device)

    set_seed(int(args.seed))

    ilsvrc_graph = os.path.join(DIR_PATH, "data/subgraphs/ilsvrc_graph")
    apy_graph = os.path.join(DIR_PATH, "data/subgraphs/apy_graph")
    awa2_graph = os.path.join(DIR_PATH, "data/subgraphs/awa2_graph")

    options = {
        "label_encoder_type": args.label_encoder_type,
        "trainval": args.trainval,
        "num_epochs": args.max_epoch,
        "batch_size": args.batch_size,
        "device": device,
        "seed": args.seed,
        "ilsvrc_graph_path": ilsvrc_graph,
        "apy_graph_path": apy_graph,
        "awa2_graph_path": awa2_graph,
    }

    # ensure save path
    if not os.path.exists(os.path.join(DIR_PATH, "save")):
        os.makedirs(os.path.join(DIR_PATH, "save"))

    model, save_path = get_label_encoder(
        options["label_encoder_type"], options
    )
    model = model.to(device)

    fc_vectors = get_fc()
    fc_vectors = fc_vectors.to(device)

    if args.label_encoder_type in ["gcn", "gat", "rgcn", "lstm", "trgcn"]:
        model, tr_log = train_gnn_model(model, fc_vectors, device, options)

        all_preds = {}
        graph_paths = [awa2_graph, apy_graph]
        with torch.no_grad():
            for i, dataset in enumerate(["awa", "apy"]):
                preds = predict(model, graph_paths[i], dataset)
                all_preds[dataset] = preds
    else:
        model, pred_obj, tr_log = train_baseline_model(
            model, fc_vectors, device
        )
        all_preds = {}
        with torch.no_grad():
            all_preds["awa"] = get_awa_preds(pred_obj)
            all_preds["apy"] = get_apy_preds(pred_obj)

    torch.save(tr_log, save_path + "_loss.json")
    torch.save(model.state_dict(), save_path + ".pt")
    torch.save(all_preds, save_path + ".pred")

    print("done!")

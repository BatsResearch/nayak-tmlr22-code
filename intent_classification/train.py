import copy
import json
import os
import random
import sys

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util as nn_util
from sklearn.metrics import accuracy_score
from zsl_kg.data.snips import SnipsDataset
from zsl_kg.example_encoders.text_encoder import TextEncoder
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG

from model.bilinear import BiLinearModel
from model.label_encoder import get_label_encoder
from utils.common import get_save_path, init_device, set_seed

INPUT_DIM = 300
HIDDEN_DIM = 32
ATTN_DIM = 20
BATCH_SIZE = 32
JOINT_DIM = 16
DECAY = 1e-05

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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

label_maps = {
    "train": ["weather", "music", "restaurant"],
    "dev": ["search", "movie"],
    "test": ["book", "playlist"],
}

concept_maps = {
    "train": ["/c/en/weather", "/c/en/music", "/c/en/restaurant"],
    "dev": ["/c/en/search", "/c/en/movie"],
    "test": ["/c/en/book", "/c/en/playlist"],
}

GLOVE_PATH = os.path.join(DIR_PATH, "data/glove.840B.300d.txt")


@click.command()
@click.option("--label_encoder_type", help="name of the label encoder")
@click.option("--seed", default=0, type=int, help="seed value")
@click.option("--lr", default=0.001, type=float)
@click.option("--decay", default=1e-05, type=float)
@click.option("--gpu", default=0, type=int, help="gpu id")
def main(label_encoder_type, seed, lr, decay, gpu):
    """
    The function is used to setup and train the model for the dataset
    and the encoder type; the function trains a bilinear model
    with a bilstm text encoder with the label encoder mentioned in the
    parameter
    """
    print("*" * 20)
    print("Training Details")
    print("DATASET: ", "snips")
    print("ENCODER: ", label_encoder_type)
    print("*" * 20)

    model_path = os.path.join(DIR_PATH, "data/models/snips")

    # create directory for saving the model
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    set_seed(seed)
    device, cuda_device = init_device(gpu)

    options = {
        "label_encoder_type": label_encoder_type,
        "lr": lr,
        "dataset": "snips",
        "joint_dim": JOINT_DIM,
        "decay": decay,
        "seed": seed,
        "graph_path": os.path.join(DIR_PATH, "data/subgraphs/snips_graph"),
        "model_path": model_path,
        "gpu": gpu,
        "device": device,
        "cuda_device": cuda_device,
    }

    # get the vocab and everything else for training the models
    model, iterator, optimizer, datasets = setup(label_encoder_type, options)

    train_epochs(model, iterator, optimizer, datasets, options, epochs=10)


def setup(label_encoder_type, options=None):
    data_path = os.path.join(DIR_PATH, "data/snips/")
    datasets = []
    for split in ["train", "dev", "test"]:
        labels = label_maps[split]
        label_to_idx = dict([(label, idx) for idx, label in enumerate(labels)])

        reader = SnipsDataset(label_to_idx)
        path = os.path.join(data_path, f"{split}.txt")
        _dataset = reader.read(path)
        datasets.append(_dataset)

    train, dev, test = datasets
    vocab = Vocabulary.from_instances(train + dev + test)

    # create the iterator
    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    print("Loading GloVe...")
    # token embed
    token_embed_path = os.path.join(DIR_PATH, "data/word_emb.pt")
    if os.path.exists(token_embed_path):
        token_embedding = torch.load(token_embed_path)
    else:
        token_embedding = Embedding.from_params(
            vocab=vocab,
            params=Params(
                {
                    "pretrained_file": GLOVE_PATH,
                    "embedding_dim": INPUT_DIM,
                    "trainable": False,
                }
            ),
        )
        torch.save(token_embedding, token_embed_path)

    print("word embeddings created...")
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    # create the text encoder
    print("Loading the text encoder...")
    text_encoder = TextEncoder(
        word_embeddings, INPUT_DIM, HIDDEN_DIM, ATTN_DIM
    )

    print("Loading the label encoder...")
    label_encoder = get_label_encoder(label_encoder_type, options=options)

    model = BiLinearModel(vocab, text_encoder, label_encoder, options=options)
    model.to(options["device"])

    optimizer = optim.Adam(
        model.parameters(), lr=options["lr"], weight_decay=options["decay"]
    )

    return model, iterator, optimizer, datasets


def get_graph(graph_path):

    train_kg = ConceptNetKG.load_from_disk(
        os.path.join(graph_path, "train_graph")
    )
    dev_kg = ConceptNetKG.load_from_disk(os.path.join(graph_path, "dev_graph"))
    test_kg = ConceptNetKG.load_from_disk(
        os.path.join(graph_path, "test_graph")
    )

    return [train_kg, dev_kg, test_kg]


def train_epochs(model, iterator, optimizer, datasets, options, epochs=10):
    """The function is used to train the model for a given number of epochs"""
    val_loss = []

    train_dataset, dev_dataset, test_dataset = tuple(datasets)
    if options["label_encoder_type"] in [
        "gcn",
        "gat",
        "rgcn",
        "lstm",
        "trgcn",
    ]:
        train_graph, dev_graph, test_graph = get_graph(options["graph_path"])
        # move to device
        train_graph.to(options["device"])
        dev_graph.to(options["device"])
        test_graph.to(options["device"])
        # get idx
        train_idx = train_graph.get_node_ids(concept_maps["train"])
        dev_idx = train_graph.get_node_ids(concept_maps["dev"])
        test_idx = train_graph.get_node_ids(concept_maps["test"])
    else:
        train_graph = None
        dev_graph = None
        test_graph = None
        wn_mapping = pd.read_csv(
            os.path.join(DIR_PATH, "misc_data/snips_mapping.csv")
        )
        graph = json.load(
            open(os.path.join(DIR_PATH, "data/induced_graph.json"), "r")
        )
        wnids = graph["wnids"]
        wnid_to_idx = dict([(wnid, idx) for idx, wnid in enumerate(wnids)])
        label_idx = [wnid_to_idx[wn_mapping["wnid"][i]] for i in range(7)]
        train_idx = label_idx[:3]
        dev_idx = label_idx[3:5]
        test_idx = label_idx[5:]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"**** Epoch {epoch} ****")
        model, optimizer = train_model(
            model,
            train_dataset,
            iterator,
            optimizer,
            train_graph,
            torch.tensor(train_idx).to(options["device"]),
        )

        loss = compute_loss(
            model,
            dev_dataset,
            iterator,
            dev_graph,
            torch.tensor(dev_idx).to(options["device"]),
        )
        print(f"Val Loss {loss}")

        if epoch > 0:
            if loss < min(val_loss):
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            best_model_wts = copy.deepcopy(model.state_dict())

        val_loss.append(loss)

        test_model(
            model,
            test_dataset,
            iterator,
            test_graph,
            torch.tensor(test_idx).to(options["device"]),
        )

    save_path = get_save_path(model.options["model_path"], options)
    torch.save(best_model_wts, save_path)

    print("done with model training")
    print("loading best model")

    model.load_state_dict(best_model_wts)
    test_model(
        model,
        test_dataset,
        iterator,
        test_graph,
        torch.tensor(test_idx).to(options["device"]),
    )

    print("done!")


def train_model(model, dataset, iterator, optimizer, train_kg, train_idx):
    """The function is used to train one epoch

    Arguments:
        model {nn.Module} -- the bilinear model
        dataset {dataset} -- the dataset loaded with allennlp loader
        iterator {Iterator} -- the bucket iterator
        cuda_device {int} -- cuda device

    Returns:
        nn.Module -- the bilinear model
    """
    model.train()
    total_batch_loss = 0.0
    generator_tqdm = Tqdm.tqdm(
        iterator(dataset, num_epochs=1, shuffle=False),
        total=iterator.get_num_batches(dataset),
    )
    loss_fn = nn.CrossEntropyLoss()

    for batch in generator_tqdm:
        # print("still here", end='\r')
        optimizer.zero_grad()
        batch = nn_util.move_to_device(batch, model.options["cuda_device"])
        output = model(batch["sentence"], train_idx, train_kg)
        labels = batch["labels"].to(model.options["device"])
        loss = loss_fn(output, labels)
        total_batch_loss += loss.item()
        loss.backward()

        # there was a nan; checking if clipping helps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

    print(f"Train loss = {total_batch_loss}")

    return model, optimizer


def compute_loss(model, dataset, iterator, kg, label_idx):
    """The function computes loss for the dataset (either train or dev).
    Based on the dev loss, we will save the model.

    Arguments:
        model {nn.Module} -- the bilinear or gile model
        dataset {nn.Dataset} -- the train or dev dataset
        iterator -- the bucket iterator with the vocab indexed

    Keyword Arguments:
        train {bool} -- indicates if we need to compute masked softmax (default: {False})

    Returns:
        float -- returns the total loss
    """
    model.eval()
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        generator_tqdm = Tqdm.tqdm(
            iterator(dataset, num_epochs=1, shuffle=False),
            total=iterator.get_num_batches(dataset),
        )
        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, model.options["cuda_device"])
            logits = model(batch["sentence"], label_idx, kg)
            loss += loss_fn(
                logits, batch["labels"].to(model.options["device"])
            )

    return loss


def test_model(model, dataset, iterator, kg, label_idx):
    """The funciton is used to test the model on the dev/test set;
    We compute the seen/unseen results and also, compute the resuts on
    only unseen classes.

    Arguments:
        model {nn.Module} -- The torch bilinear model
        dataset {dataset} -- the dataset reader
        iterator  -- the allennlp iterator; mostly this is the bucket iterator

    Returns:
        dict -- the dictionary containing all the predictions
    """
    model.eval()
    all_preds = []
    all_true = []
    all_unseen_preds = []
    output_logits = []

    # predicting on the dev/test set
    with torch.no_grad():
        generator_tqdm = Tqdm.tqdm(
            iterator(dataset, num_epochs=1, shuffle=False),
            total=iterator.get_num_batches(dataset),
        )
        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, model.options["cuda_device"])
            logits = model(batch["sentence"], label_idx, kg)
            preds = torch.argmax(logits, dim=1)
            all_true += batch["labels"].cpu().numpy().tolist()
            all_preds += preds.cpu().numpy().tolist()
            output_logits += logits.cpu().numpy().tolist()

    unsee_acc = accuracy_score(all_true, all_preds)
    print(f"unseen acc = {unsee_acc: .4f}")

    return {"unseen_acc": unsee_acc}


if __name__ == "__main__":
    main()

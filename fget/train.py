import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util as nn_util
from zsl_kg.data.fget import FineEntityTyping
from zsl_kg.example_encoders.attentive_ner import AttentiveNER

from datasets.paths import DATASETS
from evaluate import eval_model
from model.bilinear import BiLinearModel
from model.label_encoder import get_graph, get_label_encoder, get_label_idx
from utils.common import create_dirs, get_save_path, init_device, set_seed

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# TODO:
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
GLOVE_PATH = os.path.join(DIR_PATH, "data/glove.840B.300d.txt")


def setup(all_dataset, options):
    vocab_path = options["vocab_path"]

    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        vocab = Vocabulary.from_instances(all_dataset)
        vocab.save_to_files(vocab_path)

    # instantiate iterator
    iterator = BasicIterator(batch_size=1000)
    iterator.index_with(vocab)

    # load example encoder
    token_embs = Embedding.from_params(
        vocab=vocab,
        params=Params(
            {
                "pretrained_file": "(http://nlp.stanford.edu/data/glove.840B.300d.zip)#glove.840B.300d.txt",
                "embedding_dim": 300,
                "trainable": False,
            }
        ),
    )

    word_embs = BasicTextFieldEmbedder({"tokens": token_embs})
    example_encoder = AttentiveNER(
        word_embs, input_dim=300, hidden_dim=100, attn_dim=100
    )

    # load label encoder
    label_encoder = get_label_encoder(options)

    # load bilinear model
    model = BiLinearModel(
        vocab, example_encoder, label_encoder, options=options
    )

    return model, iterator


def train_model(model, datasets, iterator, options):
    # get the graphs
    if options["label_encoder_type"] in [
        "gcn",
        "gat",
        "lstm",
        "rgcn",
        "trgcn",
    ]:
        train_graph, test_graph = get_graph(options["graph_path"])
        train_graph.to(options["device"])
        test_graph.to(options["device"])
    else:
        train_graph = None
        test_graph = None

    seen_idx, unseen_idx, train_idx, test_idx = get_label_idx(
        options, test_graph
    )

    train_dataset, test_dataset = tuple(datasets)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=options["decay"]
    )
    results = []
    #
    for i in range(5):
        print("--epoch {}--".format(i + 1))
        model, optimizer, train_loss = train_epoch(
            model,
            train_dataset,
            iterator,
            optimizer,
            loss_fn,
            torch.tensor(train_idx).to(options["device"]),
            train_graph,
        )

        # TODO: evaluate the model
        print("train loss {:.4f}".format(train_loss))

        result = eval_model(
            model,
            test_dataset,
            iterator,
            torch.tensor(test_idx).to(options["device"]),
            test_graph,
            seen_idx=seen_idx,
            unseen_idx=unseen_idx,
        )

        results.append(result)

    # save results
    result_path = get_save_path(options["result_path"], options)
    result_path = result_path[:-3] + ".json"
    with open(result_path, "w+") as fp:
        json.dump(results, fp)

    # save model
    model_path = get_save_path(options["model_path"], options)
    torch.save(model.state_dict(), model_path)

    return model


def train_epoch(
    model, dataset, iterator, optimizer, loss_fn, train_idx, train_graph
):
    """The functtion trains the model for one epoch.

    Args:
        model: the bilinear odel
        dataset (str): the train dataset
        iterator (Iterator): allennlp iterator
        optimizer (nn.optimizer): Adam optimizer
        loss_fn: BCE loss

    Returns:
        tuple: tuple with the model and the optimizer
    """
    model.train()

    total_batch_loss = 0
    generator_tqdm = Tqdm.tqdm(
        iterator(dataset, num_epochs=1, shuffle=True),
        total=iterator.get_num_batches(dataset),
    )
    for batch in generator_tqdm:
        optimizer.zero_grad()
        batch = nn_util.move_to_device(batch, model.options["cuda_device"])
        logits = model(batch, train_idx, train_graph)
        labels = batch["labels"].float().to(model.options["device"])
        loss = loss_fn(logits, labels)
        total_batch_loss += loss.item()
        loss.backward()

        # checking if clipping helps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

    return model, optimizer, total_batch_loss


def load_dataset(options):
    dataset_path = options["dataset_path"]

    train_path = os.path.join(dataset_path, "clean_train.json")
    test_path = os.path.join(dataset_path, "clean_test.json")

    #
    train_df = pd.read_csv(os.path.join(dataset_path, "train_labels.csv"))
    train_labels = train_df["LABELS"].to_list()
    train_to_idx = dict(
        [(label, idx) for idx, label in enumerate(train_labels)]
    )
    train_reader = FineEntityTyping(train_to_idx)
    train_dataset = train_reader.read(train_path)

    #
    test_df = pd.read_csv(os.path.join(dataset_path, "test_labels.csv"))
    test_labels = test_df["LABELS"].to_list()
    all_labels = train_labels + test_labels
    test_to_idx = dict([(label, idx) for idx, label in enumerate(all_labels)])
    test_reader = FineEntityTyping(test_to_idx)
    test_dataset = test_reader.read(test_path)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="zero shot dataset")
    parser.add_argument("--label_encoder_type", help="label encoder type")
    parser.add_argument("--seed", default=0, type=int, help="seed no.")
    parser.add_argument("--gpu", default=0, type=int, help="gpu")
    args = parser.parse_args()

    device, cuda_device = init_device(args.gpu)

    #
    dataset_path = os.path.join(DIR_PATH, DATASETS[args.dataset]["dataset"])
    vocab_path = os.path.join(DIR_PATH, DATASETS[args.dataset]["vocab_path"])
    graph_path = os.path.join(DIR_PATH, DATASETS[args.dataset]["graph_path"])
    model_path = os.path.join(DIR_PATH, "data/models/" + args.dataset)
    result_path = os.path.join(DIR_PATH, "data/results/" + args.dataset)

    # create directories if not present
    create_dirs(model_path)
    create_dirs(result_path)

    from datasets.hyperparameters import HYP

    hyps = HYP[args.dataset]
    #
    options = {
        "seed": args.seed,
        "label_encoder_type": args.label_encoder_type,
        "dataset": args.dataset,
        "joint_dim": hyps["joint_dim"],
        "gpu": args.gpu,
        "cuda_device": cuda_device,
        "device": device,
        "dataset_path": dataset_path,
        "model_path": model_path,
        "result_path": result_path,
        "vocab_path": vocab_path,
        "graph_path": graph_path,
        "glove_path": GLOVE_PATH,
        "decay": hyps["decay"],
    }

    # set the seed
    set_seed(args.seed)

    train_dataset, test_dataset = load_dataset(options)
    all_dataset = train_dataset + test_dataset

    model, iterator = setup(all_dataset, options)

    model = model.to(device)

    print("dataset: {}".format(args.dataset))
    print("label encoder: {}".format(args.label_encoder_type))
    print("seed: {}".format(args.seed))

    # train model
    model = train_model(
        model, [train_dataset, test_dataset], iterator, options
    )

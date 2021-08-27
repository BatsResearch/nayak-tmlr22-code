import argparse
import os

import torch

from imagenet_train import get_label_encoder, pred_imagenet
from utils.common import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_encoder_type", help="label encoder")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=0, type=int)

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
        "device": device,
        "seed": args.seed,
        "ilsvrc_graph_path": ilsvrc_graph_path,
        "wordnet_graph": wordnet_graph,
    }

    model, save_path = get_label_encoder(args.label_encoder_type, options)

    model.to(device)

    model.load_state_dict(torch.load(save_path + ".pt"))

    wnids, output_vectors = pred_imagenet(model, options)
    torch.save({"wnids": wnids, "pred": output_vectors}, save_path + ".pred")

    print("saving complete!", save_path)

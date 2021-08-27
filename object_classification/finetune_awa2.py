import argparse
import json
import os
import os.path as osp

import pandas as pd
import torch
import torch.nn as nn
from IPython import embed
from scipy import io
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from zsl_kg.data.gbu import GBU

from utils.common import harmonic_mean, set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def test_on_subset(
    dataset,
    cnn,
    seen_ids,
    pred_vectors,
    all_label,
    device,
    gamma=0,
    consider_trains=False,
):
    hit = 0
    tot = 0

    loader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=False, num_workers=4
    )
    logits = torch.Tensor()

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        data = data.to(device)

        feat = cnn(data)  # (batch_size, d)
        feat = torch.cat(
            [feat, torch.ones(len(feat)).view(-1, 1).to(device)], dim=1
        )

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, seen_ids] = -1e18
        else:
            table[:, seen_ids] -= gamma

        table = table.detach().cpu()

        pred = torch.argmax(table, dim=1)
        hit += (pred == all_label).sum().item()
        tot += len(data)

        logits = torch.cat((logits, table), dim=0)

    return hit, tot, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn")
    parser.add_argument("--pred")
    parser.add_argument("--train-dir")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--consider-trains", action="store_true")
    parser.add_argument("--resnet-50", action="store_true")

    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + args.gpu)
    else:
        device = torch.device("cpu")
    print("device : ", device)

    pred_file = torch.load(args.pred, map_location="cpu")
    pred_vectors = pred_file["awa"]
    pred_vectors = pred_vectors.to(device)

    if args.resnet_50:
        cnn = resnet50(pretrained=True)
        cnn.fc = nn.Identity()

        if args.cnn:
            params = torch.load(args.cnn)
            del params["fc.weight"]
            del params["fc.bias"]
            cnn.load_state_dict(params)
    else:
        cnn = resnet101(pretrained=True)
        cnn.fc = nn.Identity()

        if args.cnn:
            params = torch.load(args.cnn)
            del params["fc.weight"]
            del params["fc.bias"]
            cnn.load_state_dict(params)

    cnn = cnn.to(device)
    cnn.eval()

    # load the fold and train/val indices
    dataset_split = io.loadmat(
        osp.join(DIR_PATH, "datasets/awa2/att_splits.mat")
    )
    image_data = pd.read_csv(
        osp.join(DIR_PATH, "datasets/awa2/image_label.csv")
    )

    all_names = []
    awa2_split = json.load(
        open(osp.join(DIR_PATH, "materials/awa2-split.json"), "r")
    )
    train_fold_names = awa2_split["train_names"]
    test_fold_names = awa2_split["test_names"]
    all_names = train_fold_names + test_fold_names
    name_to_idx = dict([(name, idx) for idx, name in enumerate(all_names)])

    seen_ids = [name_to_idx[name] for name in train_fold_names]

    # load the dataset(s)
    test_seen_loc = dataset_split["test_seen_loc"] - 1  # 0-indexed
    test_seen_loc = test_seen_loc.squeeze(-1).tolist()
    test_unseen_loc = dataset_split["test_unseen_loc"] - 1
    test_unseen_loc = test_unseen_loc.squeeze(-1).tolist()

    if args.consider_trains:
        all_test_loc = test_seen_loc + test_unseen_loc
        assert len(all_test_loc) == 5882 + 7913
    else:
        all_test_loc = test_unseen_loc
        test_seen_loc = []
        assert len(all_test_loc) == 7913

    train_labels = []
    name_to_indices = dict([(name, []) for name in all_names])
    for _id in all_test_loc:
        row = image_data.iloc[_id]
        label = row["label"]
        name_to_indices[label].append(_id)

    ave_acc = 0
    ave_acc_n = 0
    ave_seen_acc = 0.0
    ave_unseen_acc = 0.0
    seen_n = 0
    unseen_n = 0

    results = {}
    print("pred: {}".format(args.pred))
    print("num train : {}".format(len(train_fold_names)))
    print(
        "num test seen {}, num test unseen {}".format(
            len(test_seen_loc), len(test_unseen_loc)
        )
    )
    print("gamma {}".format(args.gamma))

    for i, name in enumerate(all_names, 0):
        if not name_to_indices[name]:
            continue

        dataset = GBU(
            args.train_dir,
            name_to_indices[name],
            [i] * len(name_to_indices[name]),
            stage="test",
        )
        hit, tot, logits = test_on_subset(
            dataset,
            cnn,
            seen_ids,
            pred_vectors,
            i,
            device,
            gamma=args.gamma,
            consider_trains=args.consider_trains,
        )
        acc = hit / tot
        ave_acc += acc
        ave_acc_n += 1

        if name in train_fold_names:
            ave_seen_acc += acc
            seen_n += 1
        else:
            ave_unseen_acc += acc
            unseen_n += 1

        print(
            "{} {}: {:.2f}%".format(i + 1, name.replace("+", " "), acc * 100)
        )

        # all_logits[name] = logits.cpu()
    if args.consider_trains:
        assert seen_n == 40
        assert unseen_n == 10
    else:
        assert seen_n == 0
        assert unseen_n == 10

    if args.consider_trains:
        ave_seen_acc = ave_seen_acc / seen_n
        ave_unseen_acc = ave_unseen_acc / unseen_n
        h = harmonic_mean(ave_seen_acc, ave_unseen_acc)
        print(
            "gamma: {:.2f}"
            "unseen: {:.2f} seen: {:2f} H: {} ".format(
                args.gamma, ave_unseen_acc * 100, ave_seen_acc * 100, h * 100
            )
        )
        data = {"h": h, "seen": ave_seen_acc, "unseen": ave_unseen_acc}
    else:
        ave_unseen_acc = ave_unseen_acc / unseen_n
        print("unseen {:.2f}".format(ave_unseen_acc * 100.0))
        data = {"unseen": ave_unseen_acc}

    # get path for result if gamma not present
    if not args.cnn:
        pred_file_name = os.path.basename(args.pred)
        result_path = os.path.join(DIR_PATH, f"save/awa2/{pred_file_name}")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if args.consider_trains:
            result_path = os.path.join(
                result_path, f"result_gamma_{args.gamma}.json"
            )
        else:
            result_path = os.path.join(result_path, f"result_unseen.json")

        print(f"saving result at {result_path}")
        with open(result_path, "w+") as fp:
            json.dump(data, fp)
    else:
        if args.consider_trains:
            result_path = args.cnn[:-4] + "_result.json"
        else:
            result_path = args.cnn[:-4] + "_result_unseen.json"

        print(f"saving result at {result_path}")
        with open(result_path, "w+") as fp:
            json.dump(data, fp)

    print("done!")

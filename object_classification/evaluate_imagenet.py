import argparse
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from zsl_kg.data.imagenet import ImageNet

from utils.common import pick_vectors

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def test_on_subset(dataset, cnn, n, pred_vectors, all_label, consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0

    loader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=False, num_workers=2
    )

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        data = data.cuda()

        feat = cnn(data)  # (batch_size, d)
        feat = torch.cat(
            [feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1
        )

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn")
    parser.add_argument("--resnet-50", action="store_true")
    parser.add_argument("--pred")

    parser.add_argument("--test-set")

    parser.add_argument("--output", default=None)

    parser.add_argument("--gpu", default="0")

    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--consider-trains", action="store_true")
    parser.add_argument("--test-train", action="store_true")
    parser.add_argument("--tar", action="store_true")

    args = parser.parse_args()

    test_sets = json.load(
        open(osp.join(DIR_PATH, "materials/imagenet-testsets.json"), "r")
    )
    train_wnids = test_sets["train"]
    test_wnids = test_sets[args.test_set]

    print(
        "test set: {}, {} classes, ratio={}".format(
            args.test_set, len(test_wnids), args.keep_ratio
        )
    )
    print("consider train classifiers: {}".format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file["wnids"]
    pred_vectors = pred_file["pred"]
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    ).cuda()

    pred_vectors = pred_vectors.cuda()

    n = len(train_wnids)
    m = len(test_wnids)

    if args.resnet_50:
        cnn = resnet50(pretrained=True)
        cnn.fc = nn.Identity()
        if args.cnn:
            cnn.load_state_dict(torch.load(args.cnn))
    else:
        cnn = resnet101(pretrained=True)
        if args.cnn:
            cnn.load_state_dict(torch.load(args.cnn))
        cnn.fc = nn.Identity()

    cnn = cnn.cuda()
    cnn.eval()

    TEST_TRAIN = args.test_train

    imagenet_path = os.path.join(DIR_PATH, "materials/datasets/imagenet")
    dataset = ImageNet(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda()  # top 1 2 5 10 20
    s_tot = 0

    results = {}

    if TEST_TRAIN:
        for i, wnid in enumerate(train_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(
                subset,
                cnn,
                n,
                pred_vectors,
                i - 1,
                consider_trains=args.consider_trains,
            )
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print("{}/{}, {}:".format(i, len(train_wnids), wnid), end=" ")
            for i in range(len(hits)):
                print(
                    "{:.0f}%({:.2f}%)".format(
                        hits[i] / tot * 100, s_hits[i] / s_tot * 100
                    ),
                    end=" ",
                )
            print("x{}({})".format(tot, s_tot))
    else:
        for i, wnid in enumerate(test_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(
                subset,
                cnn,
                n,
                pred_vectors,
                n + i - 1,
                consider_trains=args.consider_trains,
            )
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print("{}/{}, {}:".format(i, len(test_wnids), wnid), end=" ")
            for i in range(len(hits)):
                print(
                    "{:.0f}%({:.2f}%)".format(
                        hits[i] / tot * 100, s_hits[i] / s_tot * 100
                    ),
                    end=" ",
                )
            print("x{}({})".format(tot, s_tot))

    print("summary:", end=" ")
    for s_hit in s_hits:
        print("{:.2f}%".format(s_hit / s_tot * 100), end=" ")
    print("total {}".format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, "w"))

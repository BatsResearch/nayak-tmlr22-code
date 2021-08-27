import argparse
import json
import os
import os.path as osp

import pandas as pd
import torch
import torch.nn as nn
from scipy import io
from torchvision.models import resnet50, resnet101
from zsl_kg.data.gbu import GBU

from utils.common import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_save_path(_path, pred_file_path, resnet_50, fold):
    save_path = os.path.join(DIR_PATH, _path)
    pred_file_name = os.path.basename(pred_file_path)
    save_path = os.path.join(save_path, pred_file_name)
    if resnet_50:
        save_path = os.path.join(save_path, "resnet_50")
    else:
        save_path = os.path.join(save_path, "resnet_101")

    save_path = os.path.join(save_path, "fold_" + str(fold))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred")
    parser.add_argument("--train-dir")
    parser.add_argument("--save-path", default="save/finetune-apy")
    parser.add_argument("--resnet-50", action="store_true")
    parser.add_argument("--num-epochs", default=50, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    # setting seed value
    set_seed(0)

    # get the saved path for the fine-tuned folder
    save_path = get_save_path(
        args.save_path, args.pred, args.resnet_50, args.fold
    )

    print("save path {}".format(save_path))

    pred = torch.load(args.pred)

    # load the fold and train/val indices
    dataset_split = io.loadmat(
        osp.join(DIR_PATH, "datasets/apy/att_splits.mat")
    )
    image_data = pd.read_csv(
        osp.join(DIR_PATH, "datasets/apy/image_label.csv")
    )

    if args.fold > 0:
        with open(
            os.path.join(DIR_PATH, f"datasets/apy/trainclasses{args.fold}.txt")
        ) as fp:
            train_fold_names = [name.strip() for name in fp.readlines()]
        with open(
            os.path.join(DIR_PATH, f"datasets/apy/valclasses{args.fold}.txt")
        ) as fp:
            val_fold_names = [name.strip() for name in fp.readlines()]
    else:
        with open(
            osp.join(DIR_PATH, "datasets/apy/trainvalclasses.txt")
        ) as fp:
            train_fold_names = [name.strip() for name in fp.readlines()]

    # load the dataset(s)
    train_val_loc = dataset_split["trainval_loc"] - 1  # 0-indexed
    train_val_loc = train_val_loc.squeeze(-1).tolist()

    print("getting train/val indices")
    if args.fold > 0:
        train_name_to_idx = dict(
            [(name, idx) for idx, name in enumerate(train_fold_names)]
        )
        val_name_to_idx = dict(
            [(name, idx) for idx, name in enumerate(val_fold_names)]
        )

        train_indices = []
        train_labels = []
        val_indices = []
        val_labels = []
        for _id in train_val_loc:
            row = image_data.iloc[_id]
            label = row["label"]
            if label in train_fold_names:
                train_indices.append(_id)
                train_labels.append(train_name_to_idx[label])
            else:
                val_indices.append(_id)
                val_labels.append(val_name_to_idx[label])
    else:
        train_indices = [_id for _id in train_val_loc]
        train_name_to_idx = dict(
            [(name, idx) for idx, name in enumerate(train_fold_names)]
        )
        train_labels = []
        for _id in train_val_loc:
            row = image_data.iloc[_id]
            label = row["label"]
            train_labels.append(train_name_to_idx[label])
        val_indices = []
        val_labels = []

    print("loading the train/val dataset")
    train_dataset = GBU(
        args.train_dir, train_indices, train_labels, stage="train"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
    )

    print("total training images : {}".format(len(train_dataset)))
    print("num of classes: {}".format(len(train_fold_names)))

    # TO
    train_names = []
    with open(os.path.join(DIR_PATH, "datasets/apy/trainclasses1.txt")) as fp:
        train_names += [line.strip() for line in fp.readlines()]
    with open(os.path.join(DIR_PATH, "datasets/apy/valclasses1.txt")) as fp:
        train_names += [line.strip() for line in fp.readlines()]
    name_to_idx = dict([(name, idx) for idx, name in enumerate(train_names)])

    train_ids = [name_to_idx[name] for name in train_fold_names]
    pred_vectors = pred["apy"][train_ids, :]

    print("pred vector shape {}".format(pred_vectors.size()))

    if args.resnet_50:
        model = resnet50(pretrained=True)
    else:
        model = resnet101(pretrained=True)

    fcw = pred_vectors

    model.fc.weight = nn.Parameter(fcw[:, :-1])
    model.fc.bias = nn.Parameter(fcw[:, -1])

    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False

    model = model.cuda()
    model.train()

    #
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().cuda()

    keep_ratio = 0.9975
    trlog = {}
    trlog["loss"] = []
    trlog["acc"] = []

    for epoch in range(0, args.num_epochs):

        ave_loss = None
        ave_acc = None

        for i, (data, label) in enumerate(train_loader, 1):
            data = data.cuda()
            label = label.cuda()

            logits = model(data)
            loss = loss_fn(logits, label)

            _, pred = torch.max(logits, dim=1)
            acc = torch.eq(pred, label).type(torch.FloatTensor).mean().item()

            if i == 1:
                ave_loss = loss.item()
                ave_acc = acc
            else:
                ave_loss = ave_loss * keep_ratio + loss.item() * (
                    1 - keep_ratio
                )
                ave_acc = ave_acc * keep_ratio + acc * (1 - keep_ratio)

            print(
                "epoch {}, {}/{}, loss={:.4f} ({:.4f}), acc={:.4f} ({:.4f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    ave_loss,
                    acc,
                    ave_acc,
                )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trlog["loss"].append(ave_loss)
        trlog["acc"].append(ave_acc)

        torch.save(trlog, osp.join(save_path, "trlog"))
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                osp.join(save_path, "epoch-{}.pth".format(epoch)),
            )

    print("done!")

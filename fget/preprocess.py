## script to preprocess fine-grained entity typing datasets.

import argparse
import os

import pandas as pd

from datasets.levels import LEVELS
from datasets.paths import DATASETS
from utils.common import fine_labels, flatten_dataset, remove_labels


def split_dataset(dir_path, level, supervised):
    # load the train and test set
    train_label_path = os.path.join(dir_path, "train_labels.csv")
    test_label_path = os.path.join(dir_path, "test_labels.csv")

    for mode in ["test", "train"]:
        # read the data
        path = os.path.join(dir_path, mode + ".json")

        with open(path) as fp:
            lines = fp.readlines()

        # flatten dataset
        dataset = flatten_dataset(lines)
        if not supervised:
            if mode == "test":
                train_labels, test_labels = fine_labels(dataset, level)

                train_df = pd.DataFrame(data=train_labels, columns=["LABELS"])
                train_df.to_csv(train_label_path)

                test_df = pd.DataFrame(data=test_labels, columns=["LABELS"])
                test_df.to_csv(test_label_path)

            if mode == "train":
                # save test labels
                train_labels = pd.read_csv(train_label_path)
                dataset = remove_labels(
                    dataset, train_labels["LABELS"].to_list()
                )

            save_path = os.path.join(dir_path, "clean_" + mode + ".json")
        else:
            save_path = os.path.join(dir_path, "supervised_" + mode + ".json")

        dataset_lines = ""
        for line in dataset:
            dataset_lines += line + "\n"

        with open(save_path, "w+") as fp:
            fp.write(dataset_lines)
            fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset")
    parser.add_argument(
        "--supervised",
        help="don't remove labels if supervised",
        action="store_true",
    )
    args = parser.parse_args()

    level = LEVELS[args.dataset]
    dir_path = DATASETS[args.dataset]["dataset"]
    split_dataset(dir_path, level, args.supervised)

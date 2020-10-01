import argparse
import os

import numpy as np
import pandas as pd

from pyx.one_hot import one_hot


def add_args(parser):

    # File & directory paths
    parser.add_argument(
        "--data_dir",
        default="../dat/sim/",
        help="Path to directory from/to which to read/write data",
    )
    parser.add_argument(
        "--train_data_fname",
        default="train_labels.csv",
        help="Name of the file to which training sequences and labels will be saved",
    )
    parser.add_argument(
        "--test_data_fname",
        default="test_labels.csv",
        help="Name of the file to which test sequences and labels will be saved",
    )

    # Hyper-parameters
    parser.add_argument(
        "--validation_percentage",
        type=float,
        default=0.1,
        help="Percent of training data to split off into a validation set",
    )
    parser.add_argument(
        "-l",
        "--n_conv_layers",
        type=int,
        default=2,
        help="Number of convolutional layers to use",
    )
    parser.add_ar

    return parser


def build_cnn_model(args):
    pass


def train_model(model, train_data_loader, val_data_loader, epochs=10):
    pass


def train(args):
    assert os.path.exists(args.data_dir)

    # Load X, y into (pytorch) model-compatible form
    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_data_fname))
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_data_fname))
    train_one_hot_sequences = [
        one_hot(sequence) for sequence in train_df["sequences"].values
    ]
    test_one_hot_sequences = [
        one_hot(sequence) for sequence in test_df["sequences"].values
    ]
    train_labels = train_df[["labels_exp", "labels_out"]].values
    test_labels = test_df[["labels_exp", "labels_out"]].values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)

    train(parser.parse_args())

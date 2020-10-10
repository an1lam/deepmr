import argparse
import logging
import os
from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn

from pyx.one_hot import one_hot
from utils import detect_device


def add_args(parser):
    # Model
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Length of DNA sequences being used as input to the model",
    )
    parser.add_argument(
        "--n_conv_layers",
        type=int,
        default=1,
        help="Number of convolutional layers to use",
    )
    parser.add_argument(
        "--n_dense_layers",
        type=int,
        default=3,
        help="Number of dense (linear) layers to use",
    )
    parser.add_argument(
        "--n_outputs", type=int, default=2, help="Number of output labels to predict"
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=15,
        help="Number of filters in each convolutional layer",
    )
    parser.add_argument(
        "--filter_width",
        type=int,
        default=7,
        help="Width of each convolutional filter (aka kernel)",
    )
    parser.add_argument(
        "--dense_layer_width",
        type=int,
        default=30,
        help="Width of each dense layer (# of 'neurons')",
    )
    parser.add_argument(
        "--pooling_type",
        help="(Optional) type of pooling to use after each convolutional layer",
    )
    parser.add_argument(
        "--model_fname",
        default="cnn_counts_predictor.pt",
        help="Name of the file to save the trained model to. Typically should have .pt or .pth extension.",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of iterations to repeat full dataset optimization for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed value to override the default with."
    )

    # Data
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
        "--sequences_col",
        default="sequences",
        help="Name of column(s) containing string DNA sequences",
    )
    parser.add_argument(
        "--label_cols",
        default=["labels_exp", "labels_out"],
        help="Name of column(s) containing labels for sequences. Make sure to use the same order here as you do for testing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size used for training the model",
    )
    parser.add_argument(
        "--val_percentage",
        type=float,
        default=0.15,
        help="Fraction of training data to use to construct a validation set",
    )
    return parser


class IterablePandasDataset(torch.utils.data.IterableDataset):
    """
    One-hot encodes and returns DNA sequences and their corresponding counts.
    """

    def __init__(self, df, x_cols, y_cols=None, x_transform=None, y_transform=None):
        self.x = df[x_cols].values
        self.y = None
        if y_cols is not None:
            self.y = df[y_cols].values
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.n = len(self.x)

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(self.n):
            x = self.x[i]
            if self.y is None:
                return x, None

            y = self.y[i]
            if self.x_transform is not None:
                x = self.x_transform(x)
            if self.y_transform is not None:
                y = self.y_transform(y)
            yield x, y

    def __getitem__(self, i) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x[i]
        y = self.y[i]
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.y_transform is not None:
            y = self.y_transform(y)
        return x, y

    def __len__(self) -> int:
        return self.n


class CountsRegressor(nn.Module):
    def __init__(
        self,
        n_conv_layers,
        n_dense_layers,
        n_outputs,
        sequence_length: int,
        filters: int,
        filter_width: int,
        dense_layer_width: int,
        **kwargs,
    ):
        assert n_conv_layers >= 1 and n_dense_layers >= 1
        assert filters > 0
        super().__init__()

        conv_layers = [
            nn.Conv1d(4, filters, filter_width),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        ]
        output_size = sequence_length - filter_width + 1
        for _ in range(1, n_conv_layers):
            conv_layers.extend(
                (
                    nn.Conv1d(filters, filters, filter_width),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                )
            )
            output_size = output_size - filter_width + 1

        self.conv_layers = nn.Sequential(*conv_layers)

        dense_layers = [
            nn.Flatten(),
            nn.Linear(output_size * filters, dense_layer_width),
            nn.ReLU(),
        ]
        for _ in range(1, n_dense_layers):
            dense_layers.extend(
                (nn.Linear(dense_layer_width, dense_layer_width), nn.ReLU())
            )
        self.dense_layers = nn.Sequential(*dense_layers)
        self.regressor = nn.Linear(dense_layer_width, n_outputs)

    def forward(self, sequences, targets=None) -> dict:
        conv_output = self.conv_layers(sequences)
        dense_output = self.dense_layers(conv_output)
        predictions = self.regressor(dense_output)

        outputs = {"predictions": predictions}
        if targets is not None:
            targets = targets.float()
            loss_function = nn.MSELoss()
            loss = loss_function(predictions, targets)
            outputs["loss"] = loss

        return outputs


def run_one_epoch(
    model,
    dataloader,
    optimizer: torch.optim.Optimizer,
    training=True,
    device=None,
    metrics_config={},
):
    torch.set_grad_enabled(training)
    model.train() if training else model.eval()

    losses = []
    predictions = []
    targets = []
    metrics = {}

    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device))  # transfer data to GPU

        outputs = model(x, targets=y)  # forward pass
        predictions.extend(outputs["predictions"].detach().cpu().numpy())
        targets.extend(y.float().detach().cpu().numpy())

        if "loss" in outputs:
            loss = outputs["loss"]
            losses.append(loss.detach().cpu().numpy())
        if training:
            loss.backward()  # back propagation
            optimizer.step()
            optimizer.zero_grad()

    for name, compute_metric in metrics_config.items():
        metrics[name] = compute_metric(predictions, targets)

    return (predictions, losses, metrics)


def train_model(
    model,
    train_data_loader,
    val_data_loader,
    optimizer,
    epochs=10,
    checkpoint_fpath: Union[str, Path] = "checkpoint.hf5",
    patience=5,
    metrics_config={},
):

    patience_counter = patience
    best_avg_val_loss = np.inf
    for epoch in range(epochs):
        train_predictions, train_losses, train_metrics = run_one_epoch(
            model,
            train_data_loader,
            optimizer,
            training=True,
            metrics_config=metrics_config,
        )
        val_predictions, val_losses, val_metrics = run_one_epoch(
            model,
            val_data_loader,
            optimizer,
            training=False,
            metrics_config=metrics_config,
        )
        avg_val_loss = np.mean(val_losses)

        logging.info(
            f"Epoch {epoch}: mean training loss: {np.mean(train_losses)}, validation loss: {avg_val_loss}"
        )
        for metric_key in train_metrics.keys():
            train_metric = train_metrics[metric_key]
            val_metric = val_metrics[metric_key]
            logging.info(
                f"Epoch {epoch}: mean training {metric_key}: {train_metric}, val {metric_key}: {val_metric}"
            )

        if avg_val_loss > best_avg_val_loss:
            patience_counter -= 1
            logging.info(
                f"Epoch {epoch}: decremented patience counter to {patience_counter}"
            )
        else:
            torch.save(model.state_dict(), checkpoint_fpath)
            patience_counter = patience
            best_avg_val_loss = avg_val_loss

        if patience_counter == 0:
            model.load_state_dict(torch.load(checkpoint_fpath))
            logging.info(
                f"Epoch {epoch}: patience hit 0, reverting to best model and engaging early stopping"
            )
            break

        logging.info("*" * 50)

    # If we finished iterating but never hit early stopping, there's a chance we don't currently
    # have the best params. This ensures that we return the model loaded with the best params as
    # measured by validation loss.
    if patience_counter > 0 and patience_counter < patience:
        model.load_state_dict(torch.load(checkpoint_fpath))

    return model


def spearman_rho(x, y):
    if type(x) is list:
        x = np.array(x)
    if type(y) is list:
        y = np.array(y)
    assert x.shape == y.shape
    assert len(x.shape) <= 2

    if len(x.shape) > 0:
        return [stats.spearmanr(x[:, i], y[:, i])[0] for i in range(x.shape[1])]
    else:
        return stats.spearmanr(x, y)[0]


def pearson_r(x, y):
    if type(x) is list:
        x = np.array(x)
    if type(y) is list:
        y = np.array(y)
    assert x.shape == y.shape
    assert len(x.shape) <= 2

    if len(x.shape) > 0:
        return [stats.pearsonr(x[:, i], y[:, i])[0] for i in range(x.shape[1])]
    else:
        return stats.pearsonr(x, y)[0]


def anscombe_transform(y):
    return 2 * np.sqrt(y + 3.0 / 8)


def inverse_anscombe_transform(vals):
    return np.square(y / 2.0) - 3.0 / 8


def train(args):
    if args.seed is not None:
        np.random.seed(seed=args.seed)
        torch.manual_seed(args.seed)

    # Load training data, one-hot encode it, & split it into actual train and validation data
    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_data_fname))
    train_dataset = IterablePandasDataset(
        train_df,
        x_cols=args.sequences_col,
        y_cols=args.label_cols,
        x_transform=one_hot,
        y_transform=anscombe_transform,
    )
    n_train = len(train_dataset)
    n_val = int(n_train * args.val_percentage)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, (n_train - n_val, n_val)
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=0
    )

    # Construct counts regressor model
    device = detect_device()
    assert args.n_outputs == len(args.label_cols)
    model = CountsRegressor(**vars(args)).to(device)

    # Train the model
    model = train_model(
        model,
        train_data_loader,
        val_data_loader,
        torch.optim.Adam(model.parameters(), amsgrad=True),
        epochs=args.epochs,
        metrics_config={
            "spearman-rho": spearman_rho,
            "pearson-r": pearson_r,
        },
    )

    # Save the model to a file
    torch.save(model.state_dict(), os.path.join(args.data_dir, args.model_fname))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    train(args)

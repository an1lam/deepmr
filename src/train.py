import argparse
import logging
import os
import timeit

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from data_loader import BedPeaksDataset, load_binding_data, load_genome
from models import CNN_1d, get_big_cnn, get_default_cnn
from utils import detect_device


def test_model( test_dataset, cnn_1d,):
    """
    Instantiate, train, and return a 1D CNN model and its accuracy metrics.
    """
    # Reload data
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, num_workers=0
    )

    # Set up model and optimizer
    device = detect_device()
    cnn_1d.to(device)

    # Training loop w/ early stopping
    test_accs = []
    test_loss, test_acc = run_one_epoch(
        False, test_dataloader, cnn_1d, None, device
    )
    test_accs.append(test_acc)

    return cnn_1d, test_accs


def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):
    torch.set_grad_enabled(train_flag)
    cnn_1d.train() if train_flag else cnn_1d.eval()

    losses = []
    accuracies = []

    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device))  # transfer data to GPU

        output = cnn_1d(x)  # forward pass
        output = output.squeeze()  # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits(output, y)  # numerically stable

        if train_flag:
            loss.backward()  # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
        accuracies.append(accuracy.detach().cpu().numpy())

    return (np.mean(losses), np.mean(accuracies))

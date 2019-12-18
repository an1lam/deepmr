"""
Find instrumental variable sequence/mutation candidates for TF binding.

This program takes a convolutional neural network trained to predict TF binding and a
set of sequences and returns the most salient sequences and SNPs.
"""

import argparse
import csv
import logging
import os

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import BedPeaksDataset, load_genome, load_peak_data
from interpret import find_best_mutation
from models import get_big_cnn, get_default_cnn
from utils import INT_TO_BASES, detect_device, load_model, one_hot_decode


def _convert_to_mutation(pos_nt_pair):
    return "%d%s" % (pos_nt_pair[1], INT_TO_BASES[pos_nt_pair[0]])


def find_and_write_iv_candidates(args):
    """
    Uses a trained model to generate a list of maximally "salient" mutations to make.

    Args:
        args: Command-line args object. Expected to have attrs `data_dir` (str),
            `chroms` (list), `model_file_path` (str).
    """
    primary_model = load_model(args, args.primary_model_file_path, get_big_cnn())
    secondary_model = load_model(
        args, args.secondary_model_file_path, get_big_cnn()
    )
    secondary_model.eval()
    genome = load_genome(args.data_dir, genome_file="hg19.pkl")
    primary_data = load_peak_data(args.data_dir, args.chroms, args.binding_file)
    dataset = BedPeaksDataset(primary_data, genome, primary_model.seq_len, shuffle=True)
    data_loader = iter(torch.utils.data.DataLoader(dataset))
    salient_seqs = []

    while len(salient_seqs) < args.num_ivs:
        x_cpu, y_cpu = next(data_loader)
        x = x_cpu.to(detect_device())
        primary_model.eval()
        initial_primary_pred = (
            primary_model(x, with_sigmoid=True).squeeze().detach().cpu().numpy()
        )
        initial_secondary_pred = (
            secondary_model(x, with_sigmoid=True).squeeze().detach().cpu().numpy()
        )
        while (int(y_cpu.detach().numpy()) == 0 or initial_primary_pred < 0.5):
            x_cpu, y_cpu = next(data_loader)
            x = x_cpu.to(detect_device())
            initial_primary_pred = (
                primary_model(x, with_sigmoid=True).squeeze().detach().cpu().numpy()
            )
            initial_secondary_pred = (
                secondary_model(x, with_sigmoid=True).squeeze().detach().cpu().numpy()
            )

        # primary_model.train()
        best_mute_idx, second_best_mute_idx = find_best_mutation(
            primary_model, x_cpu, -1 if (int(y_cpu) == 1) else 1
        )
        if int(x_cpu.squeeze()[best_mute_idx[0], best_mute_idx[1]]) == 1:
            print("Most salient index matches current one. Skipping...")
            continue

        x_cpu.squeeze()[:, best_mute_idx[1]] = torch.zeros(4)
        x_cpu.squeeze()[best_mute_idx[0], best_mute_idx[1]] = 1
        x_cpu.squeeze()[:, second_best_mute_idx[1]] = torch.zeros(4)
        x_cpu.squeeze()[second_best_mute_idx[0], second_best_mute_idx[1]] = 1
        new_x = x_cpu.to(detect_device())
        new_primary_pred = (
            primary_model(new_x, with_sigmoid=True).squeeze().detach().cpu().numpy()
        )
        new_secondary_pred = (
            secondary_model(new_x, with_sigmoid=True).squeeze().detach().cpu().numpy()
        )
        print(initial_primary_pred, new_primary_pred, initial_secondary_pred, new_secondary_pred)
        salient_seqs.append(
            (
                (
                    x_cpu.squeeze().detach().numpy(),
                    _convert_to_mutation(best_mute_idx),
                    initial_primary_pred,
                    new_primary_pred,
                    initial_secondary_pred,
                    new_secondary_pred,
                )
            )
        )

    iv_file_path = os.path.join(args.data_dir, args.iv_file_path)
    with open(iv_file_path, "w", newline="") as f:
        fieldnames = [
            "sequence",
            "initial X prediction",
            "new X prediction",
            "initial Y prediction",
            "new Y prediction",
            "mutation",
        ]
        writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for (
            seq,
            mut,
            initial_primary_pred,
            new_primary_pred,
            initial_secondary_pred,
            new_secondary_pred,
        ) in salient_seqs:
            writer.writerow(
                {
                    "sequence": one_hot_decode(seq),
                    "initial X prediction": initial_primary_pred,
                    "new X prediction": new_primary_pred,
                    "initial Y prediction": initial_secondary_pred,
                    "new Y prediction": new_secondary_pred,
                    "mutation": mut,
                }
            )


if __name__ == "__main__":
    logging.getLogger("").setLevel("INFO")
    parser = argparse.ArgumentParser()
    # Directories / file names
    parser.add_argument(
        "--data_dir",
        default="../dat",
        help="(Absolute or relative) path to the directory from which we want to pull "
        "data and model files and write output.",
    )
    parser.add_argument(
        "--primary_model_file_path",
        help="Path to saved version of TF binding "
        "prediction model. Path will be appended to '--data_dir' arg."
    )
    parser.add_argument(
        "--secondary_model_file_path",
        help="Path to saved version of chromatin accessibility"
        "prediction model. Path will be appended to '--data_dir' arg."
    )
    parser.add_argument(
        "--iv_file_path",
        help="File to which we intend to write our list of sequence/SNP pairs.",
        default="iv_candidates.csv",
    )

    # IV selection configuration
    parser.add_argument(
        "-n",
        "--num_ivs",
        default=25,
        help="Number of SNPs to select as instrumental variable candidates.",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--chroms",
        nargs="+",
        default=["chr2", "chr3"],
        help="Chromosomes to use as a validation set. Typically have format "
        "'chr<number | letter (X or Y)>'.",
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="Log stuff?",
    )
    parser.add_argument("--binding_file")

    args = parser.parse_args()
    find_and_write_iv_candidates(args)

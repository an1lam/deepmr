"""
Main file for running in-silico mutagenesis experiments using simulated data.
"""

import argparse
import copy
import json
import logging
import os
from datetime import datetime

import numpy as np
from scipy.stats import bernoulli, norm
from tqdm import tqdm

from tf_coop_in_silico_mutagenesis import main as in_silico_mutagenesis_main
from tf_coop_model import main as model_main
from tf_coop_simulation import main as sim_main
from tf_coop_true_ces import main as true_ces_main


def add_args(parser):
    # Simulation
    parser.add_argument(
        "-l",
        "--sequence_length",
        type=int,
        help="Length of simulated DNA sequences",
        default=100,
    )
    parser.add_argument(
        "--exposure_motif",
        default="GATA_disc1",
        help="Name of motif for the cause TF in the simulation",
    )
    parser.add_argument(
        "--outcome_motif",
        default="TAL1_known1",
        help="Name of motif for the effect TF in the simulation",
    )
    parser.add_argument(
        "--confounder_motif",
        default=None,
        help="Name of motif for TF whose presence acts as a confounder in the simulation.",
        choices=["SOX2_1"]
    )
    parser.add_argument(
        "--confounder_prob",
        type=float,
        default=0.0,
        help="Probability of adding a non-sequence-based confounding scalar value to both the exposure and the outcome counts",
    )
    parser.add_argument(
        "--train_sequences",
        type=int,
        default=50000,
        help="Number of sequences to generate for NN training set",
    )
    parser.add_argument(
        "--test_sequences",
        type=int,
        default=50000,
        help="Number of sequences to generate for NN test set",
    )
    parser.add_argument(
        "--variant_augmentation_percentage",
        type=float,
        default=0.0,
        help="Fraction of train/test sequences to duplicate and mutate.",
    )
    parser.add_argument(
        "--max_mutations_per_variant",
        type=int,
        default=1,
        help="Maximum number of mutations to make per variant produced.",
    )

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
    parser.add_argument("--test_simdata_fname", default="test_sequences.simdata")
    parser.add_argument(
        "--train_variant_data_fname",
        default="train_variant_labels.csv",
        help="Name of the file to which training sequences and labels will be saved",
    )
    parser.add_argument(
        "--test_variant_data_fname",
        default="test_variant_labels.csv",
        help="Name of the file to which test sequences and labels will be saved",
    )
    parser.add_argument(
        "--log_summary_stats",
        action="store_true",
        help="Whether to log summary statistics about the counts at the end of the simulation.",
    )

    # Model
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
        "--model_type",
        choices=["individual", "ensemble"],
        default="individual"
    )
    parser.add_argument(
        "--n_reps",
        default=5,
        type=int,
        help="Number of ensemble components to train and save. Only used when model_type is 'ensemble'."
    )
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
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to continue training for after validation loss stops decreasing before stopping."
    )

    # Data
    parser.add_argument(
        "--train_data_fnames",
        default=[],
        action='append',
        help="Name of the file from which training sequences and labels will be loaded.",
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
    # ************************************************************************
    # In silico mutagenesis
    # ************************************************************************
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument("--results_dir_name", default=os.path.join("res", timestamp))
    parser.add_argument("--weights_dir", default="../dat/sim/ensemble")

    parser.add_argument("--exposure_name", default="GATA", choices=["GATA", "TAL1"])
    parser.add_argument("--outcome_name", default="TAL1", choices=["GATA", "TAL1"])

    # ************************************************************************
    # True causal effects computation
    # ************************************************************************
    parser.add_argument("--exposure_col", type=int, default=0)
    parser.add_argument("--outcome_col", type=int, default=1)

    # ************************************************************************
    # Experimental parameters
    # ************************************************************************
    parser.add_argument("--mixture_nonzero_prob", type=float, default=.5)
    parser.add_argument("--effect_size_ratio_mean", type=float, default=10)
    parser.add_argument("--effect_size_std", type=float, default=.5)
    parser.add_argument("--n_rounds", type=int, default=50)
    parser.add_argument("--metadata_fname", default="metadata.json")
    return parser


def sample_from_binary_mixture(p, mu, sigma, n_samples):
    choices = bernoulli.rvs(p, size=n_samples)
    locs = (mu * choices) + (1 - choices)
    return norm.rvs(loc=locs, scale=sigma, size=len(choices)), choices


def record_metadata(metadata_path, **kwargs):
    with open(metadata_path, 'w') as f:
        json.dump(kwargs, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    effect_size_samples, choices = sample_from_binary_mixture(
        args.mixture_nonzero_prob, 
        args.effect_size_ratio_mean, 
        args.effect_size_std,
        args.n_rounds
    )
    effect_size_samples = np.maximum(effect_size_samples, 0)
    alphas = norm.rvs(loc=100, scale=3, size=len(effect_size_samples))
    logging.info(f"Effect sizes: {effect_size_samples}")
    logging.info(f"Choices: {choices}")
    logging.info(f"Alphas: {alphas}")
    for i, effect_size in tqdm(enumerate(effect_size_samples)):
        exp_args = copy.deepcopy(args)
        exp_args.results_dir_name = os.path.join(exp_args.results_dir_name, f"round_{i}")
        os.makedirs(exp_args.results_dir_name, exist_ok=True)
        exp_args.alpha = alphas[i]
        exp_args.beta = alphas[i] * effect_size
        sim_main(exp_args)
        model_main(exp_args)
        in_silico_mutagenesis_main(exp_args)
        true_ces_main(exp_args)
    metadata_path = os.path.join(args.data_dir, args.results_dir_name, args.metadata_fname)
    logging.info(f"Finished all rounds. Saving metadata to {metadata_path}")
    record_metadata(
        metadata_path, 
        effect_size_samples=effect_size_samples.tolist(), 
        choices=choices.tolist(), 
        alphas=alphas.tolist()
    )

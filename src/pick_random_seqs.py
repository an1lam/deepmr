"""
Find instrumental variable sequence/mutation candidates for TF binding.

This program takes a convolutional neural network trained to predict TF binding and a
set of sequences and returns the most salient sequences and SNPs.
"""

import argparse
import logging
import os

import pandas as pd
import pybedtools

from data_loader import load_labeled_tsv_data


def pick_random_seqs(args):
    """
    Uses a trained model to generate a list of maximally "salient" mutations to make.

    Args:
        args: Command-line args object. Expected to have attrs `data_dir` (str),
            `chroms` (list), `model_file_path` (str).
    """
    tf_labels_df = load_labeled_tsv_data(
        args.data_dir, 
        args.input_bed_file_path, 
        args.chroms,
        args.verbose
    )

    tf_pos_exs_df = tf_labels_df[tf_labels_df['label'] == 'B']
    tf_neg_exs_df = tf_labels_df[tf_labels_df['label'] == 'U']

    pos_exs_sample_frac = (args.num_seqs // 2) / len(tf_pos_exs_df)
    neg_exs_sample_frac = (args.num_seqs // 2) / len(tf_neg_exs_df)
    tf_pos_exs_df = tf_pos_exs_df.sample(frac=pos_exs_sample_frac)
    tf_neg_exs_df = tf_neg_exs_df.sample(frac=neg_exs_sample_frac)
    tf_pos_exs_df = tf_pos_exs_df.drop(labels=("label"), axis=1)
    tf_neg_exs_df = tf_neg_exs_df.drop(labels=("label"), axis=1)

    target_bed_file_path = os.path.join(args.data_dir, args.output_bed_file_path)
    output_bed_tool = pybedtools.BedTool.from_dataframe(
        pd.concat((tf_pos_exs_df, tf_neg_exs_df))
    )
    output_bed_tool.saveas(target_bed_file_path)


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
        "--input_bed_file_path",
        help="Data directory relative path to .bed file from which we want to extract our sequences.",
        required=True,
    )
    parser.add_argument("--output_bed_file_path", required=True)

    # IV selection configuration
    parser.add_argument(
        "-n",
        "--num_seqs",
        default=25,
        help="Number of lines to select from the original bed file.",
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

    args = parser.parse_args()
    pick_random_seqs(args)

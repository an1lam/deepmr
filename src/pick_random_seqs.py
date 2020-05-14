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
        args: Command-line args object.
    """

    peaks_fpath = os.path.join(args.input_data_dir, args.input_bed_fname)
    tf_peaks_df = pybedtools.BedTool(fn=peaks_fpath).to_dataframe()

    sample_frac = args.num_seqs / len(tf_peaks_df)
    tf_peaks_df = tf_peaks_df.sample(frac=sample_frac)

    output_bed_fname = os.path.join(args.output_data_dir, args.output_bed_fname)
    output_bed_tool = pybedtools.BedTool.from_dataframe(tf_peaks_df[["chrom", "start", "end"]]) 
    output_bed_tool.saveas(output_bed_fname, compressed=True)


if __name__ == "__main__":
    logging.getLogger("").setLevel("INFO")
    parser = argparse.ArgumentParser()
    # Directories / file names
    parser.add_argument(
        "--input_data_dir",
        default="../dat",
        help="(Absolute or relative) path to the directory from which we want to pull "
        "data and model files and write output.",
    )
    parser.add_argument(
        "--input_bed_fname",
        help="Name of .bed file from which we want to extract our sequences.",
        required=True,
    )
    parser.add_argument(
        "--output_data_dir",
        default="../dat",
        help="(Absolute or relative) path to the directory from which we want to pull "
        "data and model files and write output.",
    )
    parser.add_argument("--output_bed_fname", required=True)

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

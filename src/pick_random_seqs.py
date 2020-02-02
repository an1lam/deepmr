"""
Find instrumental variable sequence/mutation candidates for TF binding.

This program takes a convolutional neural network trained to predict TF binding and a
set of sequences and returns the most salient sequences and SNPs.
"""

import argparse
import logging
import os

import pybedtools
from data_loader import load_bed_data


def pick_random_seqs(args):
    """
    Uses a trained model to generate a list of maximally "salient" mutations to make.

    Args:
        args: Command-line args object. Expected to have attrs `data_dir` (str),
            `chroms` (list), `model_file_path` (str).
    """
    peak_data_df = load_bed_data(args.data_dir, args.chroms, args.input_bed_file_path)
    shuffle_frac = float(args.num_seqs + 1) / len(peak_data_df)
    random_seqs_df = peak_data_df.sample(frac=shuffle_frac)
    target_bed_file_path = os.path.join(args.data_dir, args.output_bed_file_path)
    output_bed_tool = pybedtools.BedTool.from_dataframe(random_seqs_df)
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

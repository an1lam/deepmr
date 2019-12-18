import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch

from pyx.one_hot import one_hot


def load_genome(data_dir, genome_file="hg38.pkl"):
    print(f"Using genome file '{genome_file}'")
    return pickle.load(open(os.path.join(data_dir, genome_file), "rb"))


def load_bed_data(data_dir, chroms, bed_file_path, verbose=False):
    peak_data = pd.read_csv(
        os.path.join(data_dir, bed_file_path),
        sep="\t",
        names=("chrom", "start", "end", "name", "score"),
    )
    peak_data = peak_data.sort_values(by='chrom')
    if verbose:
        logging.info("First few lines of TF data:")
        logging.info(peak_data.head())
    return peak_data[peak_data["chrom"].isin(chroms)]

def load_binding_data(data_dir, chroms, bed_file_path, verbose=False):
    binding_data = pd.read_csv(
        os.path.join(data_dir, bed_file_path),
        sep="\t",
        names=("chrom", "start", "end", "name", "score"),
    )
    binding_data = binding_data.sort_values(by='chrom')
    if verbose:
        logging.info("First few lines of TF data:")
        logging.info(binding_data.head())
    return binding_data[binding_data["chrom"].isin(chroms)]


def load_accessibility_data(data_dir, chroms, bed_file_path, verbose=False):
    accessibility_data = pd.read_csv(
        os.path.join(data_dir, bed_file_path),
        sep="\t",
        names=("chrom", "start", "end", "name", "score"),
    )
    accessibility_data = accessibility_data.sort_values(by='chrom')
    if verbose:
        logging.info("First few lines of accessibility data:")
        logging.info(accessibility_data.head())
    return accessibility_data[accessibility_data["chrom"].isin(chroms)]


def load_iv_candidates(fpath):
    return pd.read_csv(fpath)


class BedPeaksDataset(torch.utils.data.IterableDataset):
    def __init__(self, binding_data, genome, context_length, shuffle=False):
        super(BedPeaksDataset, self).__init__()
        self.context_length = context_length
        self.binding_data = binding_data
        if shuffle:
            self.binding_data = binding_data.sample(frac=1)
        self.genome = genome
        self.i = 0

    def __iter__(self):
        prev_end = 0
        prev_chrom = ""
        for row in self.binding_data.itertuples():
            start, end = min(row.start, row.end), max(row.start, row.end)
            midpoint = int(0.5 * (start + end))
            seq = self.genome[row.chrom][
                midpoint
                - self.context_length // 2 : midpoint
                + self.context_length // 2
            ]
            yield (one_hot(seq), np.float32(1))  # positive example

            if prev_chrom == row.chrom:
                midpoint = int(0.5 * (prev_end + start))
                seq = self.genome[row.chrom][
                    midpoint
                    - self.context_length // 2 : midpoint
                    + self.context_length // 2
                ]
                # negative example midway inbetween peaks
                yield (one_hot(seq), np.float32(0))

            prev_chrom = row.chrom
            prev_end = end
            self.i += 1

    def __len__(self):
        return len(self.binding_data) * 2

# -*- coding: utf-8 -*-
import math
import pickle
import timeit

import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from data_loader import BedPeaksDataset
from models import CNN_1d
from pyx.one_hot import one_hot
from train import create_and_train_model

# Pytorch basics

# `pytorch` (as opposed to e.g. Theano, Tensorflow 1) uses *eager execution*: this lets you write computations as python code that you can test and debug, and later
# 1.   Backprop through (i.e. get gradients with respect to inputs)
# 2.   Run on the GPU for (hopefully!) big speedups.

if __name__ == "__main__":
    torch.manual_seed(2)  # I played with different initialization here!

    # Loading data

    # To test this out we need some data! Our task will be predict binding of
    # the important transcriptional repressor
    # [CTCF](https://en.wikipedia.org/wiki/CTCF) in a human lung cancer cell
    # line called [A549](https://en.wikipedia.org/wiki/A549_cell). The data is
    # available from ENCODE
    # [here](https://www.encodeproject.org/experiments/ENCSR035OXA/) but we
    # processed it a bit to a
    # 1. Merge replicate experiments using the "irreproducible discovery rate"
    # (IDR) [paper](https://arxiv.org/abs/1110.4705)
    # [code](https://github.com/spundhir/idr)
    # 2. Remove chromosome 1 as test data for a [kaggle
    # InClass](https://www.kaggle.com/c/predict-ctcf-binding) (more details
    # further down!)

    # Genomics data representing discrete binding events is typically stored in
    # the  `bed` format, which is described on the UCSC Genome Browser
    # [website](https://genome.ucsc.edu/FAQ/FAQformat.html#format1).

    DATADIR = "../dat/"
    binding_data = pd.read_csv(
        DATADIR + "CTCF_A549_train_2.bed",
        sep="\t",
        names=("chrom", "start", "end", "name", "score"),
    )
    binding_data[10:15]

    # `start` and `end` are positions in the genome. `name` isn't used here.
    # `score` is the `-log10(pvalue)` for a test of whether there is
    # significant binding above background. We'll ignore this for now (since
    # all these peaks are statistically significant), but you could try to
    # incorporate it in your model if you want.
    #
    # We'll split the binding data into training and validation just by taking
    # out chromosomes 2 and 3, which represents about 15% of the training data
    # we have:
    validation_chromosomes = ["chr2", "chr3"]
    train_data = binding_data[~binding_data["chrom"].isin(validation_chromosomes)]
    validation_data = binding_data[binding_data["chrom"].isin(validation_chromosomes)]
    print(len(validation_data) / len(binding_data))

    """We'll also need the human genome, which we provide here as a pickle since it's fast(ish) to load compared to reading in a txt file.

    It's worth knowing that the human genome has different *versions* that are released as more missing parts are resolved by continued sequencing and assembly efforts. Version `GRCh37` (also called `hg19`) was released in 2009, and `GRCh38` (`hg38`) was released in 2013. We'll be using `hg38`, but lots of software and data still use `hg19` (!) though so always check.
    """

    genome = pickle.load(open(DATADIR + "hg19.pkl", "rb"))

    # `genome` is just a dictionary where the keys are the chromosome
    # names and the values are strings representing the actual DNA:
    print("Example genome snippet:", genome["chr13"][100000000:100000010])

    # A substantial proportion of each chromosome is "N"s, which mark missing regions.
    print(
        "Fraction of genome made up of missingness:",
        genome["chr13"].count("N") / len(genome["chr13"]),
    )

    model = CNN_1d()
    train_dataset = BedPeaksDataset(train_data, genome, model.seq_len)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1000, num_workers=0
    )
    validation_dataset = BedPeaksDataset(validation_data, genome, model.seq_len)
    model, train_accs, val_accs = create_and_train_model(
        train_dataset, validation_dataset
    )

    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="validation")
    plt.legend()
    plt.grid(which="both")

    # cnn_1d = CNN_1d(
    #     n_output_channels=1,
    #     filter_widths=[15, 7, 5],
    #     num_chunks=5,
    #     max_pool_factor=4,
    #     nchannels=[4, 256, 512, 512],
    #     n_hidden=512,
    #     dropout=0.5,
    # )

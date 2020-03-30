import os

import numpy as np
import torch

from pyx.one_hot import one_hot

INT_TO_BASES = {0: "A", 1: "C", 2: "G", 3: "T"}
BASES = sorted(list(INT_TO_BASES.values()))


def one_hot_decode(encoded):
    """
    Returns a decoded version of a one-hot encoded NT sequence as a string.

    Args:
        encoded: A numpy array of dimensions KxL.
    """
    idx_encoded = np.argmax(encoded, axis=0)
    return "".join([INT_TO_BASES[i] for i in idx_encoded])


def detect_device():
    """Set device var based on whether a GPU is detected."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args, model_file_path, model):
    """
    Load a TF binding prediction CNN from disk.

    """
    path = os.path.join(args.data_dir, model_file_path)
    model.load_state_dict(torch.load(path))
    model.to(detect_device())
    return model


def all_kmers(k, bases=BASES):
    n_nts = len(bases)
    kmers = np.zeros(((n_nts ** k), n_nts, k), dtype=np.float64)
    i = 0

    def _all_kmers(kmer, i):
        if len(kmer) == k:
            kmers[i] = one_hot("".join(kmer))
            return i + 1

        for base in bases:
            kmer.append(base)
            i = _all_kmers(kmer, i)
            kmer.pop()

        return i

    _all_kmers([], 0)
    return kmers

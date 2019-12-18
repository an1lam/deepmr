import os

import numpy as np
import torch

INT_TO_BASES = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


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

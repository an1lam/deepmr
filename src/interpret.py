import logging

import numpy as np
import torch

from utils import detect_device


def compute_saliency(model, x_cpu):
    """
    Use a trained model to compute saliency of a batch of inputs.

    We define saliency as the gradient of the loss w.r.t. some dummy input.

    Args:
        model: A model trained to predict TF binding probabilities of one-hot encoded
            sequences.
        x_cpu: Batch of one-hot encoded sequences as torch tensors with dimensions
            'batch size (B) x x # nucleotides (K) x sequence length (L) '.
    Returns:
        The saliency of each sequence in `x_cpu` as a B-sized numpy array.
    """
    torch.set_grad_enabled(True)
    x = x_cpu.to(detect_device())
    x.requires_grad_()
    output = model(x).squeeze()
    output = torch.sigmoid(output)
    dummy = torch.ones_like(output)
    output.backward(dummy)
    gradient_np = x.grad.detach().cpu()
    logging.info("Gradients: %r", gradient_np)
    grad_mask = torch.zeros_like(x_cpu)
    saliency = gradient_np.masked_scatter_(x_cpu > 0, grad_mask).numpy()
    return saliency


def find_best_mutation(model, x_cpu, sign):
    x = x_cpu.to(detect_device())
    output = model(x).squeeze()
    # loop over all positions changing to each position nucleotide
    # note everything is implicitly parallelized over the batch here
    best_mute = None
    best_mute_effect = 0
    second_best_mute = None
    second_best_mute_effect = 0
    for seq_idx in range(model.seq_len):  # iterate over sequence
        best_out_of_nts_mute = None
        best_out_of_nts_mute_effect = 0 
        for nt_idx in range(4):  # iterate over nucleotides
            x_prime = x.clone()  # make a copy of x
            x_prime[:, :, seq_idx] = 0.0  # change the nucleotide to nt_idx
            x_prime[:, nt_idx, seq_idx] = 1.0
            output_prime = model(x_prime).squeeze()

            if sign * (output_prime - output) > (sign * best_out_of_nts_mute_effect):
                best_out_of_nts_mute_effect = output_prime - output
                best_out_of_nts_mute = (nt_idx, seq_idx)

        if sign * best_out_of_nts_mute_effect > (sign * best_mute_effect):
            best_mute_effect = best_out_of_nts_mute_effect
            best_mute = best_out_of_nts_mute
        elif sign * best_out_of_nts_mute_effect > (sign * second_best_mute_effect):
            second_best_mute_effect = best_out_of_nts_mute_effect
            second_best_mute = best_out_of_nts_mute 
    print(best_mute, second_best_mute)
    return [best_mute, second_best_mute]

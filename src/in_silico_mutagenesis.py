import argparse
import csv
from datetime import datetime
import logging
import math
import os
import pickle

import kipoi
from kipoiseq.dataloaders import SeqIntervalDl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


from custom_dropout import apply_dropout, replace_dropout_layers
from filter_instrument_candidates import filter_variants_by_score


def batches_needed(seq_len, batch_size, alpha_size=4):
    assert ((seq_len * (alpha_size)) % (batch_size)) == 0, seq_len * alpha_size
    # alpha_size - 1 mutations per nt and then account for ref in each batch
    return (seq_len * alpha_size) // (batch_size)


all_zeros = np.zeros((4,))


def generate_wt_mut_batches(seq, batch_size):
    """
    For a given sequence, generate all possible point-mutated versions of the sequence
    in batches of size `param:batch_size`.
    
    Args:
        seq (numpy.ndarray [number of base pairs, sequence length]): 
            wild type sequence.
        batch_size (int): size of returned batches. Note that each batch will have the
            wild type sequence as its first row since we need to compute wild type / mut
            prediction diffs using predictions generated by the same dropout mask.
    """
    num_nts, seq_len = seq.shape
    assert ((seq_len * num_nts) % (batch_size)) == 0, seq_len * num_nts
    n_batches = batches_needed(seq_len, batch_size, alpha_size=num_nts)
    seq_batch = seq[np.newaxis, :, :].repeat(batch_size, axis=0)
    seq_batches = seq_batch[np.newaxis, :, :, :].repeat(n_batches, axis=0)
    for seq_pos in range(seq_len):  # iterate over sequence
        for nt_pos in range(num_nts):  # iterate over nucleotides
            i = seq_pos * num_nts + nt_pos
            curr_batch, curr_idx = i // batch_size, i % (batch_size)

            curr_nt = seq[nt_pos, seq_pos]
            if int(curr_nt) == 0:
                seq_batches[curr_batch, curr_idx, :, seq_pos] = all_zeros
                seq_batches[curr_batch, curr_idx, nt_pos, seq_pos] = 1
    return seq_batches


def next_seq(it):
    return (
        np.expand_dims(next(it)["inputs"].transpose(0, 2, 1), 2)
        .astype(np.float32)
        .squeeze()
    )


def mutate_and_predict(model, seqs, epochs, batch_size, output_sel_fn=None):
    if output_sel_fn is None:
        output_sel_fn = lambda predictions: predictions

    n_seqs, n_nts, seq_len = seqs.shape
    preds = [[[] for _ in range(n_seqs)] for _ in range(epochs)]
    for i in range(n_seqs):
        seq = seqs[i]
        if np.allclose(seq, 0.25):
            raise ValueError("Bad seq: %s" % seq[:, :5])
        wt_mut_batches = generate_wt_mut_batches(seq, batch_size)
        for batch in tqdm(wt_mut_batches):
            for epoch in range(epochs):
                batch_preds = model.predict_on_batch(np.expand_dims(batch, axis=2))
                filtered_batch_preds = output_sel_fn(batch_preds)
                preds[epoch][i].append(filtered_batch_preds)

    preds = np.array(preds)
    preds = preds.reshape((epochs, n_seqs, n_nts, seq_len, -1))
    return preds


def compute_normalized_prob(train_prob):
    """
    Normalize test-time predicted probabilities based on training set prevalence.

    Source: http://deepsea.princeton.edu/help/
    """

    def normalize(prob):
        log_uniform_prob = math.log(0.05 / (1 - 0.05))
        test_log_odds = np.log(prob / (1 - prob))
        train_log_odds = np.log(train_prob / (1 - train_prob))
        denom = 1 + np.exp(log_uniform_prob - test_log_odds - train_log_odds)
        return 1 / denom

    return normalize


# Ratios and normalization formula drawn from here:
# http://deepsea.princeton.edu/media/help/posproportion.txt
def build_deepsea_normalizers(proportions_fpath):
    proportions_df = pd.read_csv(proportions_fpath, sep="\t")
    normalizers = {}

    for _, row in proportions_df.iterrows():
        cell_type = row["Cell Type"]
        treatment = row["Treatment"]
        feature = row["TF/DNase/HistoneMark"]
        key = "_".join((cell_type, feature, treatment))
        if key not in normalizers:
            norm_constant = row["Positive Proportion"]
            normalizers[key] = compute_normalized_prob(norm_constant)

    return normalizers


def compute_summary_statistics(preds, seqs, lambdas=1):
    """
    Compute summary statistics for predictions.

    Args:
        preds: np.ndarray
            Should have shape: 
            `replicates x n sequences x alphabet size x sequence length x n features`.
        seqs: np.ndarray
            A one-hot encoded representation of NT sequences.
            Should have shape: `n sequences x alphabet size x sequence length`.
    """
    n_seqs, n_nts, seq_len = seqs.shape
    epochs, n_cols = preds.shape[0], preds.shape[-1]

    means = np.mean(preds, axis=0)

    seq_idxs = seqs[np.newaxis, :].repeat(epochs, axis=0).astype(np.bool)
    ref_preds = preds[seq_idxs].reshape(epochs, n_seqs, 1, seq_len, -1)
    assert np.allclose(ref_preds[0, 0, 0, :, 0], ref_preds[0, 0, 0, 0, 0])
    mut_preds = preds[~seq_idxs].reshape(epochs, n_seqs, n_nts - 1, seq_len, -1)
    # Relies on the fact that `avg(A - B) = avg(A) - avg(B)`.
    mean_diffs = np.mean(mut_preds - ref_preds, axis=0)

    ref_vars = np.var(ref_preds, axis=0, dtype=np.float32)
    mut_vars = np.var(mut_preds, axis=0, dtype=np.float32)
    covs = np.zeros((n_seqs, n_nts - 1, seq_len, n_cols))
    for seq_idx in tqdm(range(n_seqs)):
        for seq_pos in range(seq_len):
            for col in range(n_cols):
                curr_ref_preds = ref_preds[:, seq_idx, 0, seq_pos, col]
                for nt_pos in range(n_nts - 1):
                    curr_mut_preds = mut_preds[:, seq_idx, nt_pos, seq_pos, col]
                    # Result will be a 2x2, symmetric matrix.
                    cov = np.cov(np.stack((curr_ref_preds, curr_mut_preds)), ddof=0)
                    # Get the variance (on the diagonal) and the covariance (off-diagonal).
                    assert np.allclose(cov[0, 0], ref_vars[seq_idx, 0, seq_pos, col])
                    assert np.allclose(
                        cov[1, 1], mut_vars[seq_idx, nt_pos, seq_pos, col]
                    )
                    covs[seq_idx, nt_pos, seq_pos, col] = cov[0, 1]

    stderrs = np.sqrt(lambdas * ref_vars + lambdas * mut_vars - 2 * lambdas * covs)
    return means, mean_diffs, stderrs


def get_matching_cols(all_column_names, desired_column_names):
    unique_cols = {
        label: i
        for i, label in enumerate(all_column_names)
        if label in desired_column_names
    }
    return [(v, k) for k, v in unique_cols.items()]


def filter_predictions_to_matching_cols(relevant_cols):
    def _filter(result):
        return np.array([result[:, col_idx] for col_idx, _ in relevant_cols]).T

    return _filter


def write_results(result_fpath, diffs, stderrs, x_col=0, y_col=1, sig_idxs=None):
    fieldnames = [
        "seq_num",
        "X_pred_mean",
        "X_pred_var",
        "Y_pred_mean",
        "Y_pred_var",
    ]
    if sig_idxs is None:
        sig_idxs = np.full(diffs.shape[:3], True, dtype=bool)

    with open(result_fpath, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        n_seqs, n_muts, seq_len, _ = diffs.shape
        for seq_idx in range(n_seqs):
            for seq_pos in range(seq_len):
                for nt_pos in range(n_muts):
                    if sig_idxs[seq_idx, nt_pos, seq_pos]:
                        x_eff_size = diffs[seq_idx, nt_pos, seq_pos, x_col]
                        y_eff_size = diffs[seq_idx, nt_pos, seq_pos, y_col]
                        x_stderr = stderrs[seq_idx, nt_pos, seq_pos, x_col]
                        y_stderr = stderrs[seq_idx, nt_pos, seq_pos, y_col]
                        writer.writerow(
                            {
                                "seq_num": seq_idx + 1,
                                "X_pred_mean": x_eff_size,
                                "X_pred_var": x_stderr,
                                "Y_pred_mean": y_eff_size,
                                "Y_pred_var": y_stderr,
                            }
                        )


def main(args):
    if args.override_random_seed:
        torch.manual_seed(42)

    genome_fpath = os.path.join(args.input_data_dir, args.genome_fname)
    peaks_fpath = os.path.join(args.input_data_dir, args.peaks_fname)
    dl = SeqIntervalDl(peaks_fpath, genome_fpath, auto_resize_len=args.auto_resize_len)
    data = dl.load_all()
    seqs = np.expand_dims(data["inputs"].transpose(0, 2, 1), 2).astype(np.float32)
    seqs = seqs.squeeze()
    if args.n_seqs > 0:
        seqs = seqs[: args.n_seqs]

    preds_fpath = os.path.join(args.input_data_dir, args.preds_fname)
    if args.preds_action == "write":
        deepsea = kipoi.get_model(args.kipoi_model_name, source="kipoi")
        deepsea.model = replace_dropout_layers(deepsea.model)
        deepsea.model.apply(apply_dropout)
        x_col = get_matching_cols(
            deepsea.schema.targets.column_labels, [args.x_column_name]
        )[0]
        y_col = get_matching_cols(
            deepsea.schema.targets.column_labels, [args.y_column_name]
        )[0]
        if args.verbose:
            logging.info(
                "Using '%s' Kipoi DeepSEA model for predictions", args.kipoi_model_name
            )
            logging.info("%s architecture:\n %r", args.kipoi_model_name, deepsea.model)

        n_seqs, n_nts, seq_len = seqs.shape
        logging.info(f"Generating predictions for {n_seqs} seqs")
        preds = mutate_and_predict(
            deepsea,
            seqs,
            args.epochs,
            args.batch_size,
            output_sel_fn=filter_predictions_to_matching_cols((x_col, y_col)),
        )

        proportions_fpath = os.path.join(args.input_data_dir, args.proportions_fname)
        normalizers = build_deepsea_normalizers(proportions_fpath)
        for i, (_, col_name) in enumerate((x_col, y_col)):
            preds[:, :, :, i] = normalizers[col_name](preds[:, :, :, i])

        with open(preds_fpath, "wb") as f:
            pickle.dump(preds, f)
    else:
        with open(preds_fpath, "rb") as f:
            preds = pickle.load(f)

    means, diffs, stderrs = compute_summary_statistics(preds, seqs)

    if args.verbose:
        print(f"Diffs shape: {diffs.shape}")
        print(f"Diffs: {diffs[5, 0:1, :, :]}")
    if args.results_fname:
        results_fpath = os.path.join(args.output_data_dir, args.results_fname)
        sig_var_idxs = filter_variants_by_score(diffs[:, :, :, 0])
        write_results(results_fpath, diffs, stderrs, sig_idxs=sig_var_idxs)


if __name__ == "__main__":
    logging.getLogger("").setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    # Generic
    parser.add_argument("--verbose", action="store_true", default=False)

    # Kipoi related
    parser.add_argument("--kipoi_model_name", default="DeepSEA/beluga")
    parser.add_argument("--auto_resize_len", type=int, default=2000)
    parser.add_argument(
        "--feature_column_names",
        nargs="+",
        default=["HepG2_DNase_None", "HepG2_FOXA1_None"],
    )
    parser.add_argument("--x_column_name", default="HepG2_FOXA1_None")
    parser.add_argument("--y_column_name", default="HepG2_DNase_None")

    # File paths & names
    parser.add_argument("--input_data_dir", default="../dat/deepsea/")
    parser.add_argument("--output_data_dir", default="../dat/deepsea/")
    parser.add_argument("--genome_fname", default="hg19.fa")
    parser.add_argument("--peaks_fname", default="50_random_seqs_2.bed")
    today = datetime.date(datetime.now())
    parser.add_argument("--preds_fname", default=f"predictions_{today}.pickle")
    parser.add_argument(
        "--proportions_fname", default="deepsea_normalization_constants.tsv"
    )
    parser.add_argument("--results_fname")

    # Mutagenesis related
    parser.add_argument("--preds_action", choices=["read", "write"], default="write")
    parser.add_argument("--override_random_seed", action="store_true")
    parser.add_argument("-n", "--n_seqs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=400)

    main(parser.parse_args())

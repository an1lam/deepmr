import argparse
import csv
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
import uncertainty_toolbox.viz as uviz
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.auto import tqdm, trange
from uncertainty_toolbox.recalibration import iso_recal

from ensemble import CalibratedRegressionEnsemble, Ensemble
from filter_instrument_candidates import filter_variants_by_score
from in_silico_mutagenesis import (compute_summary_statistics,
                                   generate_wt_mut_batches, write_results)
from pyx.one_hot import one_hot
from tf_coop_lib import TF_TO_MOTIF
from tf_coop_model import (CountsRegressor, IterablePandasDataset,
                           anscombe_transform, pearson_r, run_one_epoch,
                           spearman_rho)
from tf_coop_simulation import (background_frequency, simulate_counts,
                                simulate_oracle_predictions)
from utils import one_hot_decode

# %load_ext autoreload
# %autoreload 2


def add_args(parser):
    parser.add_argument(
        "--seed", type=int, help="Random seed value to override the default with."
    )
    parser.add_argument(
        "--n_conv_layers",
        type=int,
        default=3,
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
        "--sequence_length",
        type=int,
        default=100,
        help="Length of DNA sequences being used as input to the model",
    )
    parser.add_argument(
        "--model_fname",
        default="cnn_counts_predictor.pt",
        help="Name of the file to save the trained model to. Typically should have .pt or .pth extension.",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=5,
        help="Number of ensemble components to train and save. Only used when model_type is 'ensemble'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size used for training the model",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        default="../dat/sim/",
        help="Path to directory from/to which to read/write data.",
    )
    parser.add_argument("--weights_dir", default="../dat/sim/ensemble")
    parser.add_argument("--test_data_fname", default="test_labels.csv")

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument("--results_dir_name", default=os.path.join("res", timestamp))

    parser.add_argument("--sequences_col", default="sequences")
    parser.add_argument(
        "--label_cols", default=["labels_exp", "labels_out"], nargs=2,
    )
    parser.add_argument("--exposure_name", default="GATA", choices=["GATA", "TAL1"])
    parser.add_argument("--outcome_name", default="TAL1", choices=["GATA", "TAL1"])
    parser.add_argument("--confounder_motif", choices=["SOX2_1"])
    return parser



def mutate_and_predict(model, sample_dataset, predictions_key="recal_predictions"):
    preds = {}
    all_muts = []
    for seq, label in tqdm(sample_dataset):
        muts = generate_wt_mut_batches(seq, seq.shape[0] * seq.shape[1]).squeeze()
        muts = muts.transpose(0, 1, 2)
        muts = torch.from_numpy(muts)
        label = torch.from_numpy(label)
        outputs = model.predict(muts)
        preds_np = outputs[predictions_key]
        exposure_preds = preds_np[:, :, 0]
        outcome_preds = preds_np[:, :, 1]
        preds.setdefault("exposure", []).append(exposure_preds)
        preds.setdefault("outcome", []).append(outcome_preds)
        all_muts.append(muts.detach().cpu().numpy())
    return all_muts, preds


def write_results(result_fpath, diffs, stderrs, x_col=0, y_col=1, sig_idxs=None):
    fieldnames = [
        "seq_num",
        "X_pred_mean",
        "X_pred_var",
        "Y_pred_mean",
        "Y_pred_var",
    ]
    if sig_idxs is None:
        sig_idxs = np.full(diffs.shape, True, dtype=bool)

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
    # Setup
    results_dir = os.path.join(args.data_dir, args.results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # Loading data
    test_data_fpath = os.path.join(args.data_dir, args.test_data_fname)
    test_df = pd.read_csv(test_data_fpath)
    test_dataset = IterablePandasDataset(
        test_df,
        x_cols=args.sequences_col,
        y_cols=args.label_cols,
        x_transform=one_hot,
        y_transform=anscombe_transform,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0
    )

    exposure_motif_df = test_df[
        (test_df["has_exposure"] == 1)
    ]
    exposure_motif_dataset = IterablePandasDataset(
        exposure_motif_df,
        x_cols=args.sequences_col,
        y_cols=args.label_cols,
        x_transform=one_hot,
        y_transform=anscombe_transform,
    )
    exposure_motif_data_loader = torch.utils.data.DataLoader(
        exposure_motif_dataset, batch_size=args.batch_size, num_workers=0
    )

    params = {
        "n_conv_layers": args.n_conv_layers,
        "n_dense_layers": args.n_dense_layers,
        "n_outputs": args.n_outputs,
        "sequence_length": args.sequence_length,
        "filters": args.filters,
        "filter_width": args.filter_width,
        "dense_layer_width": args.dense_layer_width,
    }
    ensemble_model = Ensemble(
        args.weights_dir, args.model_fname, vars(args), n_reps=args.n_reps
    )
    calibrated_ensemble_model = CalibratedRegressionEnsemble(
        ensemble_model, test_data_loader
    )

    muts, recal_predictions = mutate_and_predict(
        calibrated_ensemble_model, exposure_motif_dataset
    )
    sample_seqs = np.array([seq for seq, label in exposure_motif_dataset])

    formatted_preds = np.stack(
        (recal_predictions["exposure"], recal_predictions["outcome"])
    )
    n_features, n_seqs, n_reps, n_variants = formatted_preds.shape
    formatted_preds = formatted_preds.transpose(2, 1, 3, 0)
    formatted_preds = formatted_preds.reshape(n_reps, n_seqs, -1, 4, n_features)
    formatted_preds = formatted_preds.transpose(0, 1, 3, 2, 4)

    means, mean_diffs, stderrs = compute_summary_statistics(
        formatted_preds, np.array(sample_seqs)
    )

    np.save(
        os.path.join(
            results_dir, f"{args.exposure_name}_{args.outcome_name}_means_calibrated.npy"
        ),
        means,
    )
    np.save(
        os.path.join(
            results_dir, f"{args.exposure_name}_{args.outcome_name}_stderrs_calibrated.npy"
        ),
        stderrs,
    )

    sig_var_idxs = filter_variants_by_score(mean_diffs[:, :, :, 0], z_threshold=1.)
    logging.info(
        "Reduced number of instruments down from %d to %d (%.2f %%)"
        % (
            np.prod(mean_diffs.shape),
            len(np.nonzero(sig_var_idxs)[0]),
            float(len(np.nonzero(sig_var_idxs)[0]) / np.prod(mean_diffs.shape)) * 100,
        )
    )
    results_fname = f"{args.exposure_name}_{args.outcome_name}_mr_inputs_calibrated.csv"
    results_fpath = os.path.join(results_dir, results_fname)
    write_results(results_fpath, mean_diffs, stderrs, sig_idxs=sig_var_idxs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import csv
import os

from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import simdna
from simdna import synthetic
import statsmodels.api as sm
import torch
from tqdm.auto import tqdm, trange
import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists_vectorized,
)
import uncertainty_toolbox.viz as uviz
from uncertainty_toolbox.recalibration import iso_recal

from ensemble import Ensemble, CalibratedRegressionEnsemble
from filter_instrument_candidates import filter_variants_by_score
from in_silico_mutagenesis import compute_summary_statistics, generate_wt_mut_batches, write_results
from pyx.one_hot import one_hot
from tf_coop_model import CountsRegressor, IterablePandasDataset
from tf_coop_model import anscombe_transform, run_one_epoch, spearman_rho, pearson_r
from tf_coop_simulation import background_frequency
from tf_coop_simulation import simulate_counts, simulate_oracle_predictions
from utils import one_hot_decode

# %load_ext autoreload
# %autoreload 2

def add_args(parser):
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed value to override the default with."
    )
    parser.add_argument(
        "--n_conv_layers",
        type=int,
        default=1,
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
        "--model_fname",
        default="cnn_counts_predictor.pt",
        help="Name of the file to save the trained model to. Typically should have .pt or .pth extension.",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        help="Number of ensemble components to train and save. Only used when model_type is 'ensemble'."
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
    parser.add_argument(
        "--weights_dir",
        default="../dat/sim/ensemble"
    )
    parser.add_argument(
        "--weights_fname",
        default="cnn_counts_predictor.pt"
    )
    parser.add_argument(
        "--test_data_fname",
        default="test_labels.csv"
    )

    parser.add_argument(
        "--results_dir",
        default="../data/sim/res"
    )

    parser.add_argument(
        "--sequences_col",
        default="sequences"
    )
    parser.add_argument(
        "--label_cols",
        default=["labels_exp", "labels_out"],
        nargs=2,
    )
    parser.add_argument( "--exposure_name", default="GATA", choices=["GATA", "TAL1"])
    parser.add_argument( "--outcome_name", default="TAL1", choices=["GATA", "TAL1"])
    parser.add_argument("--confounder_motif", choices=["SOX2_1"])
    return parser

TF_TO_MOTIF = {"GATA": "GATA_disc1", "TAL1": "TAL1_known1"}


def mutate_and_predict(model, sample_dataset, predictions_key = 'recal_predictions'):
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
        preds.setdefault('exposure', []).append(exposure_preds)
        preds.setdefault('outcome', []).append(outcome_preds)
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
    os.makedirs(args.results_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # Loading data
    test_df = pd.read_csv(args.test_data_fpath)
    test_dataset = IterablePandasDataset(
        test_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
        y_transform=anscombe_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0
    )
    both_motifs_df = test_df[(test_df['has_exposure'] == 1) & (test_df['has_outcome'] == 1)]
    both_motifs_dataset = IterablePandasDataset(
        both_motifs_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
        y_transform=anscombe_transform
    )
    both_motifs_data_loader = torch.utils.data.DataLoader(
        both_motifs_dataset, batch_size=batch_size, num_workers=0
    )

    params = {
        "n_conv_layers": args.n_conv_layers,
        "n_dense_layers": args.n_dense_layers,
        "n_outputs": args.n_outputs,
        "sequence_length": args.sequence_length,
        "filters": args.filters,
        "filter_width": args.filter_width,
        "dense_layer_width": args.dense_layer_width
    }
    ensemble_model = Ensemble(args.weights_dir, args.model_fname, params, n_reps=args.n_reps)
    calibrated_ensemble_model = CalibratedRegressionEnsemble(ensemble_model, test_data_loader)


    muts, recal_predictions = mutate_and_predict(
        calibrated_ensemble_model, both_motifs_dataset 
    )
    sample_seqs = np.array([seq for seq, label in both_motifs_sample_dataset])

    formatted_preds = np.stack((recal_predictions["exposure"], recal_predictions["outcome"]))
    n_features, n_seqs, n_reps, n_variants = formatted_preds.shape
    formatted_preds = formatted_preds.transpose(2, 1, 3, 0)
    formatted_preds = formatted_preds.reshape(n_reps, n_seqs, 4, -1, n_features)

    means, mean_diffs, stderrs = compute_summary_statistics(formatted_preds, np.array(sample_seqs))

    np.save(
        os.path.join(results_dir, f"{exposure_name}_{outcome_name}_means_calibrated_v2.npy"), 
        means
    )
    np.save(
        os.path.join(results_dir, f"{exposure_name}_{outcome_name}__stderrs_calibrated_v2.npy"),
        stderrs
    )

    sig_var_idxs = filter_variants_by_score(mean_diffs[:, :, :, 0])
    print(
        "Reduced number of instruments down from %d to %d (%.2f %%)" % 
        (np.prod(mean_diffs.shape), len(np.nonzero(sig_var_idxs)[0]), 
         float(len(np.nonzero(sig_var_idxs)[0]) / np.prod(mean_diffs.shape)) * 100)
    )
    results_fname = f'{exposure_name}_{outcome_name}_effect_sizes_calibrated_v2.csv'
    results_fpath = os.path.join(results_dir, results_fname)
    write_results(results_fpath, mean_diffs, stderrs, sig_idxs = sig_var_idxs)

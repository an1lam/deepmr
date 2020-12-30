import argparse
from datetime import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import simdna
import statsmodels.api as sm
import torch
from IPython.display import clear_output
from matplotlib import pyplot as plt
from simdna import synthetic
from tqdm.auto import tqdm

from in_silico_mutagenesis import generate_wt_mut_batches
from filter_instrument_candidates import filter_variants_by_score
from pyx.one_hot import one_hot
from tf_coop_lib import TF_TO_MOTIF
from tf_coop_model import (CountsRegressor, IterablePandasDataset,
                           anscombe_transform, pearson_r, run_one_epoch,
                           spearman_rho)
from tf_coop_simulation import (background_frequency, simulate_counts,
                                simulate_oracle_predictions)
from utils import one_hot_decode


def add_args(parser):
    parser.add_argument(
        "--seed", type=int, help="Random seed value to override the default with."
    )
    parser.add_argument(
        "--data_dir",
        default="../dat/sim/",
        help="Path to directory from/to which to read/write data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size used for loading data",
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

    parser.add_argument("--exposure_col", type=int, default=0)
    parser.add_argument("--outcome_col", type=int, default=1)
    return parser


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    exposure_motif = TF_TO_MOTIF[args.exposure_name]
    outcome_motif = TF_TO_MOTIF[args.outcome_name]

    test_data_fpath = os.path.join(args.data_dir, args.test_data_fname)
    includes_confounder = args.confounder_motif is not None

    torch.set_grad_enabled(False)

    test_df = pd.read_csv(test_data_fpath)
    test_dataset = IterablePandasDataset(
        test_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0
    )
    # both_motifs_df = test_df[(test_df['has_both'] == 1)]
    exposure_motif_df = test_df[(test_df['has_exposure'] == 1)]
    # outcome_motif_df = test_df[(test_df['has_exposure'] == 0) & (test_df['has_outcome'] == 1)]
    # neither_motif_df = test_df[
    #     (test_df['has_exposure'] == 0) & (test_df['has_outcome'] == 0)
    # ]

    # both_motifs_dataset = IterablePandasDataset(
    #     both_motifs_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
    #     y_transform=anscombe_transform,
    # )
    exposure_motif_dataset = IterablePandasDataset(
        exposure_motif_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
        y_transform=anscombe_transform
    )

    def mutate(seqs):
        preds = {}
        all_muts = []
        for seq in tqdm(seqs):
            muts = generate_wt_mut_batches(seq, seq.shape[0] * seq.shape[1]).squeeze()
            muts = muts.transpose(0, 1, 2)
            all_muts.append(muts)
            
        return np.array(all_muts)


    test_sample_seqs = [x for x, y in test_dataset]
    exposure_motif_sample_seqs = [x for x, y in exposure_motif_dataset]

    sample_seqs = np.array([seq for seq, label in exposure_motif_dataset])
    sample_labels = np.array([label for _, label in exposure_motif_dataset])
    sample_muts = mutate(sample_seqs)

    motifs = synthetic.LoadedEncodeMotifs(
        simdna.ENCODE_MOTIFS_PATH, pseudocountProb=0.001
    )
    exposure_pwm = motifs.loadedMotifs[exposure_motif].getRows()
    outcome_pwm = motifs.loadedMotifs[outcome_motif].getRows()
    confounder_pwm = None
    if args.confounder_motif is not None:
        confounder_pwm = motifs.loadedMotifs[args.confounder_motif].getRows()

    adjusted_labels_ism = []
    for i, muts in enumerate(tqdm(sample_muts)):
        adjusted_labels_ = simulate_oracle_predictions(
            [one_hot_decode(mut) for mut in muts],
            exposure_pwm,
            outcome_pwm,
            confounder_pwm=confounder_pwm,
        )
        adjusted_labels_ism.append(adjusted_labels_)

    np.array(adjusted_labels_ism).shape

    adjusted_labels_ism_no_conf = []
    if includes_confounder:
        for i, muts in enumerate(tqdm(sample_muts)):
            adjusted_labels_ = simulate_oracle_predictions(
                [one_hot_decode(mut) for mut in muts],
                exposure_pwm,
                outcome_pwm,
            )
            adjusted_labels_ism_no_conf.append(adjusted_labels_)

    adjusted_labels_ism = np.array(adjusted_labels_ism)
    adjusted_labels_ism = adjusted_labels_ism.transpose((0, 2, 1))
    adjusted_labels_ism = np.array(adjusted_labels_ism).reshape(len(sample_seqs), 4, 100, -1)
    adjusted_labels_ism_anscombe = anscombe_transform(adjusted_labels_ism)

    if includes_confounder:
        adjusted_labels_ism_no_conf = np.array(adjusted_labels_ism_no_conf)
        adjusted_labels_ism_no_conf = adjusted_labels_ism_no_conf.transpose((0, 2, 1))
        adjusted_labels_ism_no_conf = np.array(adjusted_labels_ism_no_conf).reshape(len(sample_seqs), 4, 100, -1)
        adjusted_labels_ism_no_conf_anscombe = anscombe_transform(adjusted_labels_ism_no_conf)

    seq_idxs = np.array(sample_seqs).astype(np.bool)
    adjusted_ref_labels_ism = adjusted_labels_ism_anscombe[seq_idxs].reshape(len(sample_seqs), 1, 100, -1)
    adjusted_mut_labels_ism = adjusted_labels_ism_anscombe[~seq_idxs].reshape(len(sample_seqs), 3, 100, -1)
    adjusted_diffs = adjusted_mut_labels_ism - adjusted_ref_labels_ism

    if includes_confounder:
        seq_idxs = np.array(sample_seqs).astype(np.bool)
        adjusted_ref_labels_ism_no_conf = adjusted_labels_ism_no_conf_anscombe[seq_idxs].reshape(len(sample_seqs), 1, 100, -1)
        adjusted_mut_labels_ism_no_conf = adjusted_labels_ism_no_conf_anscombe[~seq_idxs].reshape(len(sample_seqs), 3, 100, -1)
        adjusted_diffs_no_conf = adjusted_mut_labels_ism_no_conf - adjusted_ref_labels_ism_no_conf

    sig_var_idxs = filter_variants_by_score(adjusted_diffs[:, :, :, args.exposure_col])
    if includes_confounder:
        sig_var_idxs_no_conf = filter_variants_by_score(adjusted_diffs_no_conf[:, :, :, args.exposure_col])

    ols_results = []
    for i in range(len(sample_seqs)):
        if adjusted_diffs[i, sig_var_idxs[i, :, :], args.exposure_col].shape[0] > 0:
            x = adjusted_diffs[i, sig_var_idxs[i, :, :], args.exposure_col + 2].flatten()
            y = adjusted_diffs[i, sig_var_idxs[i, :, :], args.outcome_col + 2].flatten()
            ols_res = sm.OLS(y, x).fit()
            ols_results.append(ols_res)

    if includes_confounder:
        ols_results_no_conf = []
        for i in range(len(sample_seqs)):
            if adjusted_diffs_no_conf[i, sig_var_idxs_no_conf[i, :, :], args.exposure_col ].shape[0] > 0:
                x = adjusted_diffs_no_conf[i, sig_var_idxs_no_conf[i, :, :], args.exposure_col].flatten()
                y = adjusted_diffs_no_conf[i, sig_var_idxs_no_conf[i, :, :], args.outcome_col].flatten()
                ols_res = sm.OLS(y, x).fit()
                ols_results_no_conf.append(ols_res)

    ism_cis = [ols_res.params[0] for ols_res in ols_results]
    if includes_confounder:
        ism_cis_no_conf = [ols_res.params[0] for ols_res in ols_results_no_conf]

    results_dir = os.path.join(args.data_dir, args.results_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Saving true CEs to {results_dir}")
    with open(os.path.join(results_dir, f'{args.exposure_name}_{args.outcome_name}_true_ces.csv'), 'w') as f:
        f.write('seq, CI\n')
        for i, ci in enumerate(ism_cis):
            f.write(f'{i}, {ci}\n')

    if includes_confounder:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'{args.exposure_name}_{args.outcome_name}_true_ces_no_conf.csv'), 'w') as f:
            f.write('seq, CI\n')
            for i, ci in enumerate(ism_cis_no_conf):
                f.write(f'{i}, {ci}\n')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)

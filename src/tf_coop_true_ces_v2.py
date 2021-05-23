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
    parser.add_argument("--test_simdata_fname", default="test_sequences.simdata")
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument("--results_dir_name", default=os.path.join("res", timestamp))

    parser.add_argument("--sequences_col", default="sequences")
    parser.add_argument(
        "--label_cols", default=["labels_exp", "labels_out"], nargs=2,
    )
    parser.add_argument("--exposure_name", default="GATA", choices=["GATA", "TAL1"])
    parser.add_argument("--outcome_name", default="TAL1", choices=["GATA", "TAL1"])
    parser.add_argument(
        "--confounder_motif",
        default=None,
        help="Name of motif for TF whose presence acts as a confounder in the simulation.",
    )
    parser.add_argument(
        "--confounder_prob",
        type=float,
        default=0.0,
        help="Probability of adding a non-sequence-based confounding scalar value to both the exposure and the outcome counts",
    )

    parser.add_argument("--exposure_col", type=int, default=0)
    parser.add_argument("--outcome_col", type=int, default=1)
    return parser


def simdata_to_start_pos_df(simdata, exposure_motif, outcome_motif, confounder_motif=None):
    sequences = simdata.sequences
    embeddings = simdata.embeddings
    name_to_type = {
        exposure_motif: ('exposure_start_pos', 'exposure_end_pos'),
        outcome_motif: ('outcome_start_pos', 'outcome_end_pos'),
    }
    if confounder_motif is not None:
        name_to_type[confounder_motif] = ('confounder_start_pos', 'confounder_end_pos')
    motif_range_by_type = {
        'sequences': simdata.sequences,
        'exposure_start_pos': -1 * np.ones(len(embeddings), dtype=np.int),
        'outcome_start_pos': -1 * np.ones(len(embeddings), dtype=np.int),
        'confounder_start_pos': -1 * np.ones(len(embeddings), dtype=np.int),
        'exposure_end_pos': -1 * np.ones(len(embeddings), dtype=np.int),
        'outcome_end_pos': -1 * np.ones(len(embeddings), dtype=np.int),
        'confounder_end_pos': -1 * np.ones(len(embeddings), dtype=np.int)
    }
    for i, seq_embeds in enumerate(embeddings):
        for seq_embed in seq_embeds:
            keys = name_to_type[seq_embed.what.getDescription()]
            motif_len = len(seq_embed.what.string)
            motif_range_by_type[keys[0]][i] = int(seq_embed.startPos)
            motif_range_by_type[keys[1]][i] = int(seq_embed.startPos) + motif_len
    return pd.DataFrame(motif_range_by_type)


def mutate(seqs, start_pos, end_pos):
    preds = {}
    all_muts = []
    seq_fragments = [seq[:, start_pos[i]: end_pos[i]] for i, seq in enumerate(seqs)]
    for i, seq_frag in enumerate(tqdm(seq_fragments)):
        start, end = start_pos[i], end_pos[i]
        muts = generate_wt_mut_batches(seq_frag, seq_frag.shape[0] * seq_frag.shape[1]).squeeze()
        prefixes = np.repeat(np.expand_dims(seqs[i, :, :start], axis=0), len(muts), axis=0)
        suffixes = np.repeat(np.expand_dims(seqs[i, :, end:], axis=0), len(muts), axis=0)
        new_seqs = np.concatenate((prefixes, muts, suffixes), axis=2)
        all_muts.append(new_seqs)
    return np.array(all_muts)


def main(args):
    torch.set_grad_enabled(False)
    if args.seed is not None:
        np.random.seed(args.seed)

    exposure_motif = TF_TO_MOTIF[args.exposure_name]
    outcome_motif = TF_TO_MOTIF[args.outcome_name]

    logging.info(f"Loading from {args.data_dir}")
    test_data_fpath = os.path.join(args.data_dir, args.test_data_fname)
    raw_simulation_data_fpath = os.path.join(args.data_dir, args.test_simdata_fname)
    includes_confounder = (args.confounder_motif is not None) or (args.confounder_prob > 0)

    simdata = synthetic.read_simdata_file(raw_simulation_data_fpath )
    start_pos_df = simdata_to_start_pos_df(simdata, exposure_motif, outcome_motif, confounder_motif=args.confounder_motif)
    start_pos_df.head()

    test_df = pd.read_csv(test_data_fpath)
    test_df = pd.merge(test_df, start_pos_df, on="sequences")

    test_dataset = IterablePandasDataset(
        test_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0
    )
    exposure_motif_df = test_df[(test_df['has_exposure'] == 1)]
    exposure_motif_dataset = IterablePandasDataset(
        exposure_motif_df, x_cols=args.sequences_col, y_cols=args.label_cols, x_transform=one_hot,
        y_transform=anscombe_transform
    )

    test_sample_seqs = [x for x, y in test_dataset]
    exposure_motif_sample_seqs = [x for x, y in exposure_motif_dataset]

    sample_seqs = np.array([seq for seq, label in exposure_motif_dataset])
    sample_labels = np.array([label for _, label in exposure_motif_dataset])
    sample_ranges = exposure_motif_df[['exposure_start_pos', 'exposure_end_pos']].values
    start_pos, end_pos = sample_ranges[:, args.exposure_col], sample_ranges[:, args.outcome_col]
    sample_seq_fragments = [seq[:, start_pos[i]: end_pos[i]] for i, seq in enumerate(sample_seqs)]
    sample_muts = mutate(sample_seqs, start_pos, end_pos)

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
            confounder_pwm=None
        )
        adjusted_labels_ism.append(adjusted_labels_)

    fragment_length = end_pos[0] - start_pos[0]
    assert np.allclose(fragment_length, end_pos - start_pos)
    adjusted_labels_ism = np.array(adjusted_labels_ism)
    adjusted_labels_ism = adjusted_labels_ism.transpose((0, 2, 1))
    adjusted_labels_ism = np.array(adjusted_labels_ism).reshape(len(sample_seqs), fragment_length, 4, -1)
    adjusted_labels_ism = adjusted_labels_ism.transpose((0, 2, 1, 3))
    adjusted_labels_ism_anscombe = anscombe_transform(adjusted_labels_ism)
    seq_idxs = np.array(sample_seq_fragments).astype(np.bool)
    adjusted_ref_labels_ism = adjusted_labels_ism_anscombe[seq_idxs].reshape(len(sample_seqs), 1, fragment_length, -1)
    adjusted_mut_labels_ism = adjusted_labels_ism_anscombe[~seq_idxs].reshape(len(sample_seqs), 3, fragment_length, -1)
    adjusted_diffs = adjusted_mut_labels_ism - adjusted_ref_labels_ism
    assert np.all(adjusted_ref_labels_ism[0, 0, :, 2] == adjusted_ref_labels_ism[0, 0, 0, 2])

    ols_results = []
    for i in range(len(sample_seqs)):
        x = adjusted_diffs[i, :, :, args.exposure_col + 2].flatten()
        y = adjusted_diffs[i, :, :, args.outcome_col + 2].flatten()
        assert len(x) == fragment_length * 3, len(x)
        assert len(y) == fragment_length * 3, len(y)
        ols_res = sm.OLS(y, x).fit()
        ols_results.append(ols_res)

    ism_cis = [ols_res.params[0] for ols_res in ols_results]

    results_dir = os.path.join(args.data_dir, args.results_dir_name)

    logging.info(f"Saving true CEs to {results_dir}")
    with open(os.path.join(results_dir, f'{args.exposure_name}_{args.outcome_name}_true_ces.csv'), 'w') as f:
        f.write('seq, CI\n')
        for i, ci in enumerate(ism_cis):
            f.write(f'{i}, {ci}\n')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)

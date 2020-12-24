import csv
import os

from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
import seaborn as sns
import simdna
from simdna import synthetic
import statsmodels.api as sm
import torch
from tqdm.auto import tqdm, trange

from utils import one_hot_decode
from in_silico_mutagenesis import compute_summary_statistics, generate_wt_mut_batches, write_results
from pyx.one_hot import one_hot
from tf_coop_model import CountsRegressor, IterablePandasDataset
from tf_coop_model import anscombe_transform, run_one_epoch, spearman_rho, pearson_r
from tf_coop_simulation import background_frequency
from tf_coop_simulation import simulate_counts, simulate_oracle_predictions

np.random.seed(42)

n_conv_layers = 3
n_dense_layers = 3
n_outputs = 2
sequence_length = 100
filters = 15
filter_width = 7
dense_layer_width = 30

data_dir = '../dat/sim_conf'
weights_dir = os.path.join(data_dir, 'ensemble')
weights_fname = 'cnn_counts_predictor_with_variants.pt'
test_data_fpath = os.path.join(data_dir, 'test_labels.csv')
raw_simulation_data_fpath = os.path.join(data_dir, 'test_sequences.simdata')
mutagenesis_results_dir = os.path.join(data_dir, 'res')

os.makedirs(mutagenesis_results_dir, exist_ok=True)

sequences_col = "sequences"
label_cols = ["labels_exp", "labels_out"]
batch_size = 1000
n_samples = 10

includes_confounder = True
exposure_motif = "GATA_disc1"
outcome_motif = "TAL1_known1"
confounder_motif = "SOX2_1" if includes_confounder else None

torch.set_grad_enabled(False)


class Ensemble:
    def __init__(self, model_base_dir, model_fname, model_params, n_reps=5):
        models = []
        for i in range(1, n_reps+1):
            model = CountsRegressor(**params)
            model.load_state_dict(torch.load(os.path.join(model_base_dir, str(i), model_fname)))
            models.append(model)
        self.models = models

    def predict(self, seqs, targets=None):
        preds = {}
        for model in self.models:
            model_preds = model(seqs, targets=targets)
            for key, preds_ in model_preds.items():
                preds.setdefault(key, []).append(preds_.detach().cpu().numpy())
        return {k: np.stack(v) for k, v in preds.items()}


params = {
    "n_conv_layers": n_conv_layers,
    "n_dense_layers": n_dense_layers,
    "n_outputs": n_outputs,
    "sequence_length": sequence_length,
    "filters": filters,
    "filter_width": filter_width,
    "dense_layer_width": dense_layer_width
}
ensemble_model = Ensemble(weights_dir, "cnn_counts_predictor.pt", params, n_reps=5)
ensemble_model

# # Load Test Data & Generate Predictions
# Now we're going to load test data to get some basic metrics about how well our model performs.

test_df = pd.read_csv(test_data_fpath)
test_dataset = IterablePandasDataset(
    test_df, x_cols=sequences_col, y_cols=label_cols, x_transform=one_hot,
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, num_workers=0
)

# # Running Deep MR
# In Shrikumar et al, all effects are computed in raw counts space. Here, for purposes of making our result relevant to Deep MR, we compute interaction effects in both Anscombe-transformed space and raw counts space but focus on the validity of the former.

both_motifs_df = test_df[(test_df['has_exposure'] == 1) & (test_df['has_outcome'] == 1)]
exposure_motif_df = test_df[(test_df['has_exposure'] == 1) & (test_df['has_outcome'] == 0)]
outcome_motif_df = test_df[(test_df['has_exposure'] == 0) & (test_df['has_outcome'] == 1)]
neither_motif_df = test_df[
    (test_df['has_exposure'] == 0) & (test_df['has_outcome'] == 0)
]

len(both_motifs_df), len(exposure_motif_df), len(outcome_motif_df), len(neither_motif_df)

both_motifs_dataset = IterablePandasDataset(
    both_motifs_df, x_cols=sequences_col, y_cols=label_cols, x_transform=one_hot,
    y_transform=anscombe_transform
)
both_motifs_data_loader = torch.utils.data.DataLoader(
    both_motifs_dataset, batch_size=batch_size, num_workers=0
)


def mutate_and_predict(model, sample_dataset):
    preds = {}
    all_muts = []
    for seq, label in tqdm(sample_dataset):
        muts = generate_wt_mut_batches(seq, seq.shape[0] * seq.shape[1]).squeeze()
        muts = muts.transpose(0, 1, 2)
        muts = torch.from_numpy(muts)
        label = torch.from_numpy(label)
        preds_ = model.predict(muts, targets=label)['predictions']
        exposure_preds = preds_[:, :, 0]
        outcome_preds = preds_[:, :, 1]
        preds.setdefault('exposure', []).append(exposure_preds)
        preds.setdefault('outcome', []).append(outcome_preds)
        all_muts.append(muts.detach().cpu().numpy())
    return all_muts, preds


# both_motifs_sample_dataset, _ = torch.utils.data.random_split(
#     both_motifs_dataset, (n_samples, len(both_motifs_dataset) - n_samples)
# )
both_motifs_sample_dataset = both_motifs_dataset

muts, preds = mutate_and_predict(ensemble_model, both_motifs_dataset)
sample_seqs = [seq for seq, label in both_motifs_dataset]

# +
exposure_col = "exposure"
outcome_col = "outcome"

formatted_preds = np.stack((preds[exposure_col], preds[outcome_col]))
n_features, n_seqs, n_reps, n_variants = formatted_preds.shape
formatted_preds = formatted_preds.transpose(2, 1, 3, 0)
formatted_preds = formatted_preds.reshape(n_reps, n_seqs, 4, -1, n_features)

means, mean_diffs, stderrs = compute_summary_statistics(formatted_preds, np.array(sample_seqs))

np.save(os.path.join(mutagenesis_results_dir, "GATA_TAL1_means_v6.npy"), means)
np.save(os.path.join(mutagenesis_results_dir, "GATA_TAL1_stderrs_v6.npy"), stderrs)


# +
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


# -

from filter_instrument_candidates import filter_variants_by_score
sig_var_idxs = filter_variants_by_score(mean_diffs[:, :, :, 0])
print(
    "Reduced number of instruments down from %d to %d (%.2f %%)" % 
    (np.prod(mean_diffs.shape), len(np.nonzero(sig_var_idxs)[0]), 
     float(len(np.nonzero(sig_var_idxs)[0]) / np.prod(mean_diffs.shape)) * 100)
)
print(sig_var_idxs.shape)
results_fname = f'GATA_TAL1_effect_sizes_v3.csv'
results_fpath = os.path.join(mutagenesis_results_dir, results_fname)
write_results(results_fpath, mean_diffs, stderrs, sig_idxs = sig_var_idxs)

ols_results = []
for i in trange(len(sample_seqs)):
    if mean_diffs[i, sig_var_idxs[i, :, :], 0].shape[0] > 0:
        x = sm.add_constant(mean_diffs[i, sig_var_idxs[i, :, :], 0].flatten(), prepend=False)
        y = mean_diffs[i, sig_var_idxs[i, :, :], 1].flatten()
        ols_res = sm.OLS(y, x).fit()
        ols_results.append(ols_res)

seq_cis = [r.params[0] for r in ols_results]

sns.distplot(seq_cis)

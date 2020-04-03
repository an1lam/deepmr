import numpy as np
from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F

from np_utils import abs_max
from utils import all_kmers
from utils import detect_device
from utils import one_hot_decode
from utils import INT_TO_BASES


BASES = list(INT_TO_BASES.values())


def mut_effects_to_impact_map(seq, mut_effects):
    K, L = seq.shape
    assert mut_effects.shape[0] == K - 1
    impact_map = np.zeros_like(seq)

    i = 0
    for l in range(L):  # iterate over sequence
        for k in range(K):  # iterate over nucleotides
            curr_nt = seq[k, l]
            if int(curr_nt) == 0:
                mek, mel = i % (K - 1), i // (K - 1)
                impact_map[k, l] = mut_effects[mek, mel]
                i += 1

    return impact_map


def batch_conv_score(targets, filters, score_function, batches=4 ** 4):
    """
    Args:
        targets: B x K x L numpy array representing the targets of convolution.
        filters: N x K x M numpy array representing conv filters.
    """
    B, K, L = targets.shape
    assert filters.shape[1] == K
    N, M = filters.shape[0], filters.shape[2]
    device = detect_device()

    assert N % batches == 0, "%d not divisible by %d" % (N, batches)
    filter_batches = np.split(filters, batches)

    filter_scores = np.zeros(N)
    targets_ = torch.from_numpy(targets).to(device)
    for i, filter_batch in enumerate(tqdm(filter_batches, desc='Convolving filters')):
        batch_size = filter_batch.shape[0]
        filter_batch_ = torch.from_numpy(filter_batch).to(device)
        raw_scores = F.conv1d(targets_, filter_batch_).cpu().numpy()
        batch_filter_scores = score_function(raw_scores)
        assert (i + 1) * batch_size <= N, "%d vs. %d" % ((i + 1) * batch_size, N)
        filter_scores[i * batch_size : (i + 1) * batch_size] = batch_filter_scores

    return filter_scores


# def logo_stuff():
#     for i, impact_map in tqdm(enumerate(impact_maps)):
#         maxvals, maxidxs = impact_map.max(), impact_map.idxmax()
#         center_on_idx = maxidxs[maxvals.idxmax()]
#         start_idx = max(center_on_idx - context_size, 0)
#         end_idx = min(center_on_idx + context_size + 1, len(impact_map))
#         scaled_seq_df = pd.DataFrame(seqs[i].T / 2, columns=["A", "C", "T", "G"])
#         seq_logo = Logo(scaled_seq_df.iloc[start_idx: end_idx, :])
#         seq_logo.fig.savefig(f'../seq_logo_{i}.png')
#         impact_logo = Logo(impact_map.iloc[start_idx: end_idx, :])
#         impact_logo.fig.savefig(f'../dat/impact_map_logo_{i}.png')


def kmer_mut_scores(seqs, mut_effects, pwms, max_k=12):
    def compute_agg_score(raw_scores):
        return np.sum(abs_max(raw_scores, axis=2), axis=0)

    B, K, L = seqs.shape
    impact_maps = np.zeros(seqs.shape)
    for i, seq in enumerate(seqs):
        impact_maps[i] = mut_effects_to_impact_map(seq, mut_effects[i])

    ks = set(min(pwm.shape[1], max_k) for pwm in pwms)
    kmers_by_k = {}
    for k in tqdm(ks, desc="Generate kmers"):
        kmers_by_k[k] = all_kmers(k)

    kmer_scores_by_pwm = {one_hot_decode(pwm): [] for pwm in pwms}
    for i, pwm in enumerate(tqdm(pwms, desc="Weighted scoring")):
        k = min(pwm.shape[1], max_k)
        kmers = kmers_by_k[k]
        scores = batch_conv_score(
            impact_maps, kmers, compute_agg_score,
            batches=max(4 ** (k - 7), 1)
        )
        kmer_scores_by_pwm[one_hot_decode(pwm)].append((kmers, scores))
    return kmer_scores_by_pwm


def kmer_pwm_scores(pwms, max_k=12):
    def compute_pwm_score(raw_scores):
        assert raw_scores.shape[0] == 1
        return abs_max(raw_scores, axis=2)[0]

    ks = set(min(pwm.shape[1], max_k) for pwm in pwms)
    kmers_by_k = {}
    for k in tqdm(ks, desc="Generate kmers"):
        kmers_by_k[k] = all_kmers(k)

    kmer_scores_by_pwm = {one_hot_decode(pwm): [] for pwm in pwms}
    for i, pwm in enumerate(tqdm(pwms, desc="PWM scoring")):
        kmers = kmers_by_k[min(pwm.shape[1], max_k)]

        scores = batch_conv_score(
            np.expand_dims(pwm, axis=0), kmers, compute_pwm_score,
            batches=max(4 ** (k - 5), 1)
        )
        kmer_scores_by_pwm[one_hot_decode(pwm)].append((kmers, scores))
    return kmer_scores_by_pwm


def top_n_kmer_mut_scores(seqs, mut_effects, pwms, n=20, max_k=12):
    kmer_scores_by_pwm = kmer_mut_scores(seqs, mut_effects, pwms, max_k=max_k)
    top_kmers_scores = {one_hot_decode(pwm): [] for pwm in pwms}
    for pwm, kmer_scores in kmer_scores_by_pwm.items():
        for kmers, scores in kmer_scores:
            top_idxs = np.argsort(-1 * scores)
            top_scores = scores[top_idxs][:n]
            top_kmers = kmers[top_idxs, :, :][:n]
            top_kmers_scores[pwm].append(list(zip(top_kmers, top_scores)))
    return top_kmers_scores


def top_n_kmer_pwm_scores(pwms, n=20, max_k=12):
    kmer_scores_by_pwm = kmer_pwm_scores(pwms, max_k=max_k)
    top_kmers_scores = {one_hot_decode(pwm): [] for pwm in pwms}
    for pwm, kmer_scores in kmer_scores_by_pwm.items():
        for kmers, scores in kmer_scores:
          top_idxs = np.argsort(-1 * scores)
          top_scores = scores[top_idxs][:n]
          top_kmers = kmers[top_idxs, :, :][:n]
          top_kmers_scores[pwm].append(list(zip(top_kmers, top_scores)))
    return top_kmers_scores

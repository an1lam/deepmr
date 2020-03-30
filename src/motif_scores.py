import numpy as np
from tqdm import tqdm
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


def compute_kmer_weighted_scores(impact_maps, kmers, batches=4 ** 4):
    """
    Args:
        impact_map: B x K x L numpy array representing the effect of (K-1)*L mutations
            on DeepSEA's FOXA1 binding prediction.
        kmers: N x K x M numpy array representing kmers.
    """
    B, K, L = impact_maps.shape
    assert kmers.shape[1] == K
    N, M = kmers.shape[0], kmers.shape[2]
    device = detect_device()

    assert N % batches == 0, "%d not divisible by %d" % (N, batches)
    kmer_batches = np.split(kmers, batches)

    kmer_scores = np.zeros(N)
    impact_maps_ = torch.from_numpy(impact_maps).to(device)
    for i, kmer_batch in enumerate(kmer_batches):
        batch_size = kmer_batch.shape[0]
        kmer_batch_ = torch.from_numpy(kmer_batch).to(device)
        raw_scores = F.conv1d(impact_maps_, kmer_batch_).cpu().numpy()
        batch_kmer_scores = np.sum(abs_max(raw_scores, axis=2), axis=0)
        kmer_scores[i * batch_size : (i + 1) * batch_size] = batch_kmer_scores

    return kmer_scores


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
    B, K, L = seqs.shape
    impact_maps = np.zeros(seqs.shape)
    for i, seq in enumerate(seqs):
        impact_maps[i] = mut_effects_to_impact_map(seq, mut_effects[i])

    ks = set(pwm.shape[1] for pwm in pwms)
    kmers_by_k = {}
    for k in tqdm(ks, desc='Generate kmers'):
        kmers_by_k[k] = all_kmers(k)

    kmer_scores_by_pwm = {}
    for i, pwm in enumerate(tqdm(pwms, desc='Weighted scoring')):
        k = min(pwm.shape[1], max_k)
        kmers = kmers_by_k[k]
        scores = compute_kmer_weighted_scores(
            impact_maps,
            kmers,
            batches=max(4 ** (k - 5), 1)
        )
        assert one_hot_decode(pwm) not in kmer_scores_by_pwm
        kmer_scores_by_pwm[one_hot_decode(pwm)] = (kmers, scores)
    return kmer_scores_by_pwm


def kmer_pwm_scores(pwms, max_k=12):
    ks = set(pwm.shape[1] for pwm in pwms)
    kmers_by_k = {}
    for k in tqdm(ks, desc='Generate kmers'):
        kmers_by_k[k] = all_kmers(k)

    kmer_scores_by_pwm = {}
    for i, pwm in enumerate(tqdm(pwms, desc='PWM scoring')):
        kmers = kmers_by_k[pwm.shape[1]]
        scores = np.sum(pwm * kmers, axis=(1, 2))
        kmer_scores_by_pwm[one_hot_decode(pwm)] = (kmers, scores)
    return kmer_scores_by_pwm


def top_n_kmer_mut_scores(seqs, mut_effects, pwms, n=20, max_k=12):
    kmer_scores_by_pwm = kmer_mut_scores(seqs, mut_effects, pwms, max_k=max_k)
    top_kmers_scores = {}
    for pwm, (kmers, scores) in kmer_scores_by_pwm.items():
        top_idxs = np.argsort(-1 * scores)
        top_scores = scores[top_idxs][:n]
        top_kmers = kmers[top_idxs, :, :][:n]
        top_kmers_scores[pwm] = list(zip(top_kmers, top_scores))
    return top_kmers_scores


def top_n_kmer_pwm_scores(pwms, n=20, max_k=12):
    kmer_scores_by_pwm = kmer_pwm_scores(pwms, max_k=max_k)
    top_kmers_scores = {}
    for pwm, (kmers, scores) in kmer_scores_by_pwm.items():
        top_idxs = np.argsort(-1 * scores)
        top_scores = scores[top_idxs][:n]
        top_kmers = kmers[top_idxs, :, :][:n]
        top_kmers_scores[pwm] = list(zip(top_kmers, top_scores))
    return top_kmers_scores

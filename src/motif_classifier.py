import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.nn import functional as F
from utils import motif_to_pwm


class MotifClassifier:
    def __init__(self, motifs):
        self.motifs = motifs
        max_motif_len = max(motif.length for motif in motifs)
        pwms = np.array([
            motif_to_pwm(motif, as_log=True, pad_to_len=max_motif_len)
            for motif in self.motifs
        ])
        self.pwms = torch.from_numpy(pwms)

        self.trained = False
        self.pwm_thresholds = {}

    def train(self, seqs, labels):
        if type(seqs) is np.ndarray:
            seqs = torch.from_numpy(seqs)

        scores_by_pwm = {motif.consensus: [] for motif in self.motifs}
        labels_by_pwm = {motif.consensus: [] for motif in self.motifs}
        for seq, label in zip(seqs, labels):
            highest_score, highest_scoring_pwm = self._predict(seq, self.pwms)
            key = self.motifs[highest_scoring_pwm].consensus
            scores_by_pwm[key].append(highest_score)
            labels_by_pwm[key].append(label)

        for i, motif in enumerate(self.motifs):
            key = motif.consensus
            assert len(labels_by_pwm[key]) == len(scores_by_pwm[key])
            if len(labels_by_pwm[key]) > 0:
                fpr, tpr, thresholds = roc_curve(
                    labels_by_pwm[key], scores_by_pwm[key]
                )
                j = np.argmax(tpr + (1 - fpr))
                self.pwm_thresholds[key] = thresholds[j]

        self.trained = True

    @classmethod
    def _predict(cls, seq, pwms):
        scores, _ = F.conv1d(seq.unsqueeze(0), pwms)[0].max(axis=1)
        highest_scoring_pwm = scores.argmax()
        highest_score = scores.max()
        return highest_score.numpy(), highest_scoring_pwm.numpy()

    def __call__(self, seq):
        if not self.trained:
            raise ValueError(
                "Shouldn't call PWMClassifier in test mode without training it."
            )

        if type(seq) is np.ndarray:
            seq = torch.from_numpy(seq)

        highest_score, highest_scoring_pwm = self._predict(seq, self.pwms)
        threshold = self.pwm_thresholds[self.motifs[highest_scoring_pwm].consensus]
        return highest_score, (highest_score >= threshold).astype(int)

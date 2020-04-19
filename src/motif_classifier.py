import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.nn import functional as F
from utils import motif_to_pwm


class MotifClassifier:
    def __init__(self, motifs):
        self.motifs = motifs
        pwms = np.array(
            [motif_to_pwm(motif, as_log=True, as_torch=False) for motif in self.motifs]
        )
        self.pwms = torch.from_numpy(pwms)

        self.trained = False
        self.pwm_thresholds = {}

    def train(self, seqs, labels):
        if type(seqs) is np.ndarray:
            seqs = torch.from_numpy(seqs)

        scores_by_pwm = {motif.consensus: [] for motif in self.motifs}
        labels_by_pwm = {motif.consensus: [] for motif in self.motifs}
        for seq_batch, label_batch in zip(seqs, labels):
            highest_scores, highest_scoring_pwms = self._predict_batch(
                seq_batch, self.pwms
            )
            for i in range(len(seq_batch)):
                j = highest_scoring_pwms[i]
                key = self.motifs[j].consensus
                scores_by_pwm[key].append(highest_scores[i])
                labels_by_pwm[key].append(label_batch[i])

        for i, motif in enumerate(self.motifs):
            key = motif.consensus
            fpr, tpr, thresholds = roc_curve(
                labels_by_pwm[key], scores_by_pwm[key]
            )
            j = np.argmax(tpr + (1 - fpr))
            self.pwm_thresholds[key] = thresholds[j]

        self.trained = True

    @classmethod
    def _predict_batch(cls, seqs, pwms):
        scores, _ = F.conv1d(seqs, pwms).max(axis=2)
        highest_scoring_pwms = scores.argmax(axis=1)
        highest_scores, _ = scores.max(axis=1)
        return highest_scores.numpy(), highest_scoring_pwms.numpy()

    def __call__(self, seqs):
        if not self.trained:
            raise ValueError(
                "Shouldn't call PWMClassifier in test mode without training it."
            )

        if type(seqs) is np.ndarray:
            seqs = torch.from_numpy(seqs)

        highest_scores, highest_scoring_pwms = self._predict_batch(seqs, self.pwms)
        thresholds = np.array([
            self.pwm_thresholds[self.motifs[i].consensus]
            for i in highest_scoring_pwms
        ])
        return (highest_scores >= thresholds).astype(int)

import unittest

from Bio.motifs import Motif
import numpy as np

from motif_classifier import MotifClassifier
from pyx.one_hot import one_hot
from utils import BASES


class MotifClassifierTest(unittest.TestCase):
    def test_train_one_batch(self):
        counts = {"A": [100] * 4}
        counts["C"] = [50] * 4
        counts["G"] = counts["T"] = [0] * 4
        motif = Motif(counts=counts)
        model = MotifClassifier([motif])
        seqs = np.array([one_hot("A" * 8), one_hot("C" * 8)])
        labels = np.array([1, 0])
        seqs = np.expand_dims(seqs, axis=0)
        labels = np.expand_dims(labels, axis=0)

        model.train(seqs, labels)
        np.testing.assert_array_equal(model(seqs.squeeze()), labels.squeeze())


if __name__ == "__main__":
    unittest.main()

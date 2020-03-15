import unittest

import numpy as np

from pyx.one_hot import one_hot
from align import prob_sw


class ProbSWTest(unittest.TestCase):

    def test_prob_sw(self):
        pwm = one_hot("AGC")
        seqs = np.expand_dims(one_hot("AGCT"), axis=0)

        score, end_idx = prob_sw(pwm, seqs)
        self.assertEqual(3, score)
        self.assertEqual(2, end_idx[0], end_idx)

    def test_prob_sw__seq_gaps(self):
        pwm = one_hot("AGTC")
        seqs = np.expand_dims(one_hot("AGCTCAAA"), axis=0)
        print("****", seqs.shape)

        score, end_idx = prob_sw(pwm, seqs)
        self.assertEqual(3.95, score[0])
        self.assertEqual(4, end_idx[0], end_idx)

    def test_prob_sw__pwm_gaps(self):
        pwm = one_hot("GCT")
        seqs = np.expand_dims(one_hot("CTAA"), axis=0)

        score, end_idx = prob_sw(pwm, seqs)
        self.assertEqual(2, score[0])
        self.assertEqual(1, end_idx[0], end_idx)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import TestCase

import numpy as np

from motif_scores import batch_conv_score
from motif_scores import mut_effects_to_impact_map
from motif_scores import top_n_kmer_mut_scores
from motif_scores import top_n_kmer_pwm_scores
from np_utils import abs_max
from pyx.one_hot import one_hot
from utils import all_kmers

def _agg_score(raw_scores):
    return np.sum(abs_max(raw_scores, axis=2), axis=0)

class TestMutEffectsToImpactMap(TestCase):
    def test_success(self):
        seq = one_hot("ACGT")
        mut_effects = np.zeros((3, 4))
        mut_effects[2, 0] = 1
        mut_effects[0, 1:] = 1

        expected = one_hot("TAAA")
        result = mut_effects_to_impact_map(seq, mut_effects)

        np.testing.assert_array_equal(expected, result)

    def test_regression_case(self):
        seq = np.array((one_hot("AAAA")))
        mut_effects = np.zeros((3, 4))
        mut_effects[1, :] = 1

        expected = one_hot("GGGG")
        result = mut_effects_to_impact_map(seq, mut_effects)

        np.testing.assert_array_equal(expected, result)


class TestBatchedConvScores(TestCase):
    def test_one_impact_map_one_batch(self):
        kmers = all_kmers(3)
        impact_map = np.zeros((4, 4))
        impact_map[3, :] = 0.5
        impact_maps = np.expand_dims(impact_map, axis=0)

        result = batch_conv_score(impact_maps, kmers, _agg_score, batches=1)

        self.assertEqual((64), len(result))
        self.assertEqual(1.5, np.max(result))
        self.assertEqual(0, np.min(result))
        self.assertTrue(np.any(result == 1.0))

    def test_multiple_impact_maps_one_batch(self):
        kmers = all_kmers(3)
        impact_map = np.zeros((4, 4))
        impact_map[3, :] = 0.5
        impact_maps = np.stack((impact_map, impact_map), axis=0)

        result = batch_conv_score(impact_maps, kmers, _agg_score, batches=1)

        self.assertEqual((64,), result.shape)
        self.assertEqual(3.0, np.max(result))

    def test_multiple_impact_maps_multiple_batchs(self):
        kmers = all_kmers(3)
        impact_map = np.zeros((4, 4))
        impact_map[3, :] = 0.5
        impact_maps = np.stack((impact_map, impact_map), axis=0)

        result = batch_conv_score(impact_maps, kmers, _agg_score, batches=1)

        self.assertEqual((64,), result.shape)
        self.assertEqual(3.0, np.max(result))


class TestTopNKmerMutScores(TestCase):
    def test_one_score_one_k(self):
        seqs = np.array((one_hot("AAAA"), one_hot("TTTT")))
        mut_effects_1 = np.zeros((3, 4))
        mut_effects_1[1, :] = 1
        mut_effects_2 = np.zeros((3, 4))
        mut_effects_2[0, :] = 1
        mut_effects = np.stack((mut_effects_1, mut_effects_2))
        pwms = np.array((one_hot("GGG"),))

        result = top_n_kmer_mut_scores(seqs, mut_effects, pwms, n=1)

        self.assertEqual(1, len(result))
        self.assertEqual(1, len(result["GGG"]))
        self.assertEqual(1, len(result["GGG"][0]))
        # Note: relying on arbitrary property of sort.
        # Could just as well be 'GGG' or one of 6 other 3-mers.
        np.testing.assert_array_equal(one_hot("AAA"), result["GGG"][0][0][0])
        self.assertEqual(3, result["GGG"][0][0][1])

    def test_one_score_one_k_max_k(self):
        seqs = np.array((one_hot("AAAA"), one_hot("TTTT")))
        mut_effects_1 = np.zeros((3, 4))
        mut_effects_1[1, :] = 1
        mut_effects_2 = np.zeros((3, 4))
        mut_effects_2[0, :] = 1
        mut_effects = np.stack((mut_effects_1, mut_effects_2))
        pwms = np.array((one_hot("GGG"),))

        result = top_n_kmer_mut_scores(seqs, mut_effects, pwms, n=1, max_k=2)

        self.assertEqual(1, len(result))
        self.assertEqual(1, len(result["GGG"]))
        self.assertEqual(1, len(result["GGG"][0]))
        # Note: relying on arbitrary property of sort.
        # Could just as well be 'GGG' or one of 6 other 3-mers.
        np.testing.assert_array_equal(one_hot("AA"), result["GGG"][0][0][0])
        self.assertEqual(2, result["GGG"][0][0][1])

    def test_multiple_scores_one_k(self):
        seqs = np.array((one_hot("AAAA"), one_hot("TTTT")))
        mut_effects_1 = np.zeros((3, 4))
        mut_effects_1[1, :] = 1
        mut_effects_2 = np.zeros((3, 4))
        mut_effects_2[0, :] = 1
        mut_effects = np.stack((mut_effects_1, mut_effects_2))
        pwms = np.array((one_hot("GGG"), one_hot("TTT")))

        result = top_n_kmer_mut_scores(seqs, mut_effects, pwms, n=2)

        self.assertEqual(2, len(result))
        self.assertEqual(1, len(result["GGG"]))
        self.assertEqual(2, len(result["GGG"][0]))
        self.assertEqual(3, result["GGG"][0][0][1])
        self.assertEqual(3, result["GGG"][0][1][1])
        np.testing.assert_array_equal(one_hot("AAA"), result["GGG"][0][0][0])
        np.testing.assert_array_equal(one_hot("GAA"), result["GGG"][0][1][0])


class TestTopNKmerPWMScores(TestCase):
    def test_one_score_two_pwms(self):
        pwms = np.array((one_hot("GGG"), one_hot("TTT")), dtype=np.float64)

        result = top_n_kmer_pwm_scores(pwms, n=1)

        self.assertEqual(2, len(result))
        self.assertEqual(1, len(result["GGG"]))
        np.testing.assert_array_equal(one_hot("GGG"), result["GGG"][0][0][0])
        self.assertEqual(3, result["GGG"][0][0][1])

    def test_one_score_one_pwm_max_k(self):
        pwms = np.array((one_hot("GGG"), one_hot("TTT")), dtype=np.float64)

        result = top_n_kmer_pwm_scores(pwms, n=1, max_k=2)

        self.assertEqual(2, len(result))
        self.assertEqual(1, len(result["GGG"]))
        np.testing.assert_array_equal(one_hot("GG"), result["GGG"][0][0][0])
        self.assertEqual(2, result["GGG"][0][0][1])

if __name__ == "__main__":
    unittest.main()

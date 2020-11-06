import unittest

import numpy as np

from tf_coop_simulation import background_frequency
from tf_coop_simulation import ddg_pwm_score
from tf_coop_simulation import simulate_counts



class TestPwmScore(unittest.TestCase):
    def test_background_pwm(self):
        pwm = np.array([background_frequency[nt] for nt in "ACTG"])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1  # All As

        scores = ddg_pwm_score(sequences, pwm)

        # All log-space scores will be 0.
        np.testing.assert_array_equal(
            scores.shape, (1, 6)
        )  # L - K + 1 for 1D convolution
        np.testing.assert_allclose(scores, 0.5)

    def test_near_perfectly_matched_pwm(self):
        pwm = np.array([0.97, 0.01, 0.01, 0.01])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1  # All As

        scores = ddg_pwm_score(sequences, pwm)

        # Computed manually using Wolfram Alpha as a sanity check:
        #
        #   https://www.wolframalpha.com/input/?i=1%2F%281%2Bexp%28-+log%28.97+%2F+.27%29+*+5%29%29
        np.testing.assert_allclose(scores, 0.9983, rtol=0.001)

    def test_single_mutation_impact(self):
        pwm = np.array([0.997, 0.001, 0.001, 0.001])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1  # All As

        ref_scores = ddg_pwm_score(sequences, pwm)
        mutants = sequences.copy()
        # First A -> C so only impacts PWM application #1
        mutants[0, 0, 0] = 0
        mutants[0, 1, 0] = 1
        mut_scores = ddg_pwm_score(mutants, pwm)
        # Computed manually using Wolfram Alpha as a sanity check:
        #
        #   https://bit.ly/313OREG
        np.testing.assert_almost_equal(np.sum(ref_scores - mut_scores), .5515367)

    def test_perfect_confidence_not_allowed(self):
        pwm = np.array([1.0, 0.0, 0.0, 0.0])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1  # All As

        with self.assertRaises(AssertionError):
            ddg_pwm_score(sequences, pwm)



class TestSimulateCounts(unittest.TestCase):
    def test_background_sequence(self):
        pwm = np.array([background_frequency[nt] for nt in "ACTG"])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = ["A" * 100]

        q_exp, q_out = simulate_counts(sequences, pwm, pwm)

        # Probability at any position is .5 so almost guaranteed across 100 positions
        np.testing.assert_almost_equal(q_exp, 1.)
        np.testing.assert_almost_equal(q_out, 1.)

    def test_perfectly_matching_and_nonmatching_sequence(self):
        pwm_exp = np.array([0.97, 0.01, 0.01, 0.01])[:, None]
        pwm_exp = np.repeat(pwm_exp, 5, axis=1).T
        pwm_out = np.array([0.001, 0.997, 0.001, 0.001])[:, None]
        pwm_out = np.repeat(pwm_out, 5, axis=1).T
        sequences = ["A" * 100]
        q_exp, q_out = simulate_counts(sequences, pwm_exp, pwm_out)

        # See PWM test above for Wolfram Alpha score computation link
        np.testing.assert_almost_equal(q_exp, 1.0)
        np.testing.assert_almost_equal(q_out, 0)



if __name__ == "__main__":
    unittest.main()

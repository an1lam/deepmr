import unittest

import numpy as np

from tf_coop_simulation import background_frequency
from tf_coop_simulation import ddg_pwm_score


class TestPwmScore(unittest.TestCase):

    def test_background_pwm(self):
        pwm = np.array([background_frequency[nt] for nt in 'ACTG'])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1 # All As

        scores = ddg_pwm_score(sequences, pwm)

        # All log-space scores will be 0.
        np.testing.assert_allclose(scores, 0.5)

    def test_near_perfectly_matched_pwm(self):
        pwm = np.array([.97, 0.01, 0.01, 0.01])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1 # All As

        scores = ddg_pwm_score(sequences, pwm)

        # Computed manually using Wolfram Alpha as a sanity check:
        #
        #   https://www.wolframalpha.com/input/?i=1%2F%281%2Bexp%28-+log%28.97+%2F+.27%29+*+5%29%29
        np.testing.assert_allclose(scores, 0.9983, rtol=.001)

    def test_perfect_confidence_not_allowed(self):
        pwm = np.array([1., 0., 0., 0.])[:, None]
        pwm = np.repeat(pwm, 5, axis=1).T
        sequences = np.zeros((1, 4, 10))
        sequences[0, 0, :] = 1 # All As

        with self.assertRaises(AssertionError):
            ddg_pwm_score(sequences, pwm)



if __name__ == "__main__":
    unittest.main()

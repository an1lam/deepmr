import numpy as np
import unittest
from unittest import TestCase

from pyx.one_hot import one_hot
from utils import all_kmers
from utils import BASES


class AllKmersTest(TestCase):
    def test_k_equals_1(self):
        onemers = all_kmers(1)
        expected = [one_hot(base) for base in BASES]
        np.testing.assert_array_equal(expected, onemers)

    def test_k_equals_4(self):
        fourmers = all_kmers(4)
        expected_len = 256
        self.assertEqual(expected_len, fourmers.shape[0])


if __name__ == "__main__":
    unittest.main()

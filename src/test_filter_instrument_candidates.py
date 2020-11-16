import unittest

import numpy as np
from scipy import stats

from filter_instrument_candidates import filter_variants_by_score


class FilterVariantsByScoreTest(unittest.TestCase):

    def test_filter_normally_distributed_effects(self):
        veffs = stats.norm.rvs(size=1000)
        idxs = filter_variants_by_score(veffs)
        filtered_veffs = veffs[idxs]
        frac = float(len(filtered_veffs) / len(veffs))
        self.assertTrue((np.abs(filtered_veffs) >= np.min(np.abs(veffs[~idxs]))).all())
        self.assertTrue(frac <= .07, frac)


if __name__ == "__main__":
    unittest.main()
        

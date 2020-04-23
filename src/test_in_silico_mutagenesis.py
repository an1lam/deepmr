from mock import MagicMock
import unittest

import numpy as np

from pyx.one_hot import one_hot
from in_silico_mutagenesis import mutate_and_predict


class MutateAndPredictTest(unittest.TestCase):
    def test_mutate_and_predict_one_batch_one_epoch(self):
        seqs = np.expand_dims(one_hot("ACGTA"), axis=0)

        model = MagicMock()
        preds_shape = seqs.shape + (1,)
        ordered_vals = np.arange(np.prod(preds_shape)).reshape(preds_shape)
        model.predict_on_batch = MagicMock(return_value=ordered_vals)

        preds = mutate_and_predict(
            model, seqs, 1, np.prod(seqs.shape)
        )

        self.assertEqual((1, 1, 4, 5, 1), preds.shape)
        np.testing.assert_array_equal(preds[0], ordered_vals)


if __name__ == "__main__":
    unittest.main()

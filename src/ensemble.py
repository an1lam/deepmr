import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import torch
import uncertainty_toolbox.data as udata
import uncertainty_toolbox.metrics as umetrics
from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists_vectorized,
)
import uncertainty_toolbox.viz as uviz
from uncertainty_toolbox.recalibration import iso_recal

from utils import one_hot_decode
from tf_coop_model import CountsRegressor, IterablePandasDataset


class Ensemble:
    def __init__(self, model_base_dir, model_fname, model_params, model_cls = CountsRegressor, n_reps=5, n_features = 2):
        models = []
        for i in range(1, n_reps+1):
            model = model_cls(**model_params)
            model.load_state_dict(torch.load(os.path.join(model_base_dir, str(i), model_fname)))
            models.append(model)
        self.models = models
        self.n_features = n_features
        self.n_reps = n_reps

    def predict(self, seqs, targets=None):
        preds = {}
        for model in self.models:
            model_preds = model(seqs, targets=targets)
            for key, preds_ in model_preds.items():
                preds.setdefault(key, []).append(preds_.detach().cpu().numpy())
            if 'predictions' in preds:
                preds['mean'] = np.mean(preds['predictions'], axis=0)
                preds['std'] = np.std(preds['predictions'], axis=0)

        return {k: np.stack(v) for k, v in preds.items()}
    

class CalibratedRegressionEnsemble:
    def __init__(self, base_model, calibration_data_loader):
        self._base_model= base_model
        self._recalibrators = self.fit_recalibrators(base_model, calibration_data_loader)
        
    @classmethod
    def fit_recalibrators(cls, model, data_loader):
        batch_size = data_loader.batch_size
        predictions = np.zeros((len(model.models), len(data_loader.dataset), model.n_features))
        ys = np.zeros((len(data_loader.dataset), model.n_features))
        for (i, (x, y)) in enumerate(data_loader):
            assert len(y) == batch_size, f"len(y) = {len(y)} vs. batch size = {batch_size}"
            p = model.predict(x)['predictions']
            predictions[:, i * batch_size: (i + 1) * batch_size, :] = p
            ys[i * batch_size: (i + 1) * batch_size, :] = y.numpy()
            
        pred_means = np.mean(predictions, axis=0).squeeze()
        pred_stds = np.std(predictions, axis=0).squeeze()
        
        recal_models = []
        for f in range(pred_means.shape[1]):
            y = ys[:, f]
            pred_mean, pred_std = pred_means[:, f], pred_stds[:, f]
            exp_props, obs_props = get_proportion_lists_vectorized(pred_mean, pred_std, y)
            recal_model = iso_recal(exp_props, obs_props)
            recal_models.append(recal_model)
        return recal_models

    def predict(self, seqs, targets=None):
        preds = self._base_model.predict(seqs, targets=targets)
        # shape: E x B x F
        predictions = preds['predictions']
        recal_predictions = np.zeros_like(preds['predictions'])
        pred_means = np.mean(predictions, axis=0)
        pred_stds = np.std(predictions, axis=0)
        
        for f in range(self._base_model.n_features):
            pred_dist = stats.norm(loc=pred_means[:, f], scale=pred_stds[:, f])
            for c in range(self._base_model.n_reps):
                recal_model = self._recalibrators[f]
                orig_preds = predictions[c, :, f]
                orig_quantiles = pred_dist.cdf(orig_preds)
                recal_quantiles = recal_model.predict(orig_quantiles)
                recal_predictions[c, :, f] = pred_dist.ppf(recal_quantiles)
        preds['recal_predictions'] = recal_predictions
        return preds

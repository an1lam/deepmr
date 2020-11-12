"""
Functionality to filter a set of candidate instruments (variants) by their
(variant) effect sizes into a subset of "strong" instrument candidates.

To do so, we pretend that variant effects are normally distributed and
only keep variants (instruments) with |z-score| greater than some threshold.
"""

import numpy as np

def filter_variants_by_score(variant_effects, z_threshold=2.):
    """
    Take the subset of variant effects with z-score greater than the specified threshold.

    Args:
        variant_effects: np.ndarray
            A numpy tensor of arbitrary shape.
        variant_stds: np.ndarray
            A numpy tensor with the same shape as `variant_effects`.
        z_threshold: float
            A positive float representing the minimum absolute z-score of retained variants.

    Returns:
        Tuple[np.ndarray, np.ndarray]
           Indices for variants which satisfy the threshold requirement.
    """

    standardized_effects = np.abs(
        (variant_effects - np.mean(variant_effects)) / np.std(variant_effects)
    )
    return (standardized_effects >= z_threshold)


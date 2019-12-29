import argparse
import logging
import math
import os
import torch

import numpy as np
import pandas as pd
import scipy.special
import sklearn.linear_model
from matplotlib import pyplot as plt

import statsmodels.api as sm
from data_loader import load_iv_candidates
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def logistic_regress_y_on_x(X, y):
    y = (y > 0.5).astype(np.float32)
    X = sm.add_constant(X)

    lr = sm.Logit(y, X)
    result = lr.fit()

    return result


def regress_y_on_x(X, y):
    X = sm.add_constant(X)

    lr = sm.OLS(y, X)
    result = lr.fit()

    return result


def compute_wald_ratio(iv_candidate_df):
    tf_y_neg_exs = iv_candidate_df["initial X prediction"].values
    tf_y_pos_exs = iv_candidate_df["new X prediction"].values
    acc_y_neg_exs = iv_candidate_df["initial Y prediction"].values
    acc_y_pos_exs = iv_candidate_df["new Y prediction"].values

    X_neg_exs = np.zeros_like(tf_y_neg_exs)
    X_pos_exs = np.ones_like(tf_y_pos_exs)

    X = np.concatenate((X_neg_exs, X_pos_exs))
    tf_y = np.concatenate((tf_y_neg_exs, tf_y_pos_exs))
    acc_y = np.concatenate((acc_y_neg_exs, acc_y_pos_exs))
    x_on_z_result = regress_y_on_x(X, tf_y)
    y_on_z_result = regress_y_on_x(X, acc_y)
    y_on_z_coef = y_on_z_result.params[1]
    x_on_z_coef = x_on_z_result.params[1]
    y_on_z_se = y_on_z_result.bse[1]
    x_on_z_se = x_on_z_result.bse[1]
    wald_ratio = y_on_z_result.params[1] / x_on_z_result.params[1]
    wald_ratio_se = math.sqrt(
        (y_on_z_se ** 2) / (x_on_z_coef ** 2)
        + ((y_on_z_coef ** 2) * (x_on_z_se ** 2)) / (x_on_z_coef ** 4)
    )
    return wald_ratio, wald_ratio_se, x_on_z_result.pvalues + y_on_z_result.pvalues


def plot_logistic_regression_result(result, X, y, samples=10):
    X_samples = np.linspace(X.min(), X.max(), samples)
    predictions = result.predict(sm.add_constant(X_samples))
    plt.scatter(X, y)
    plt.plot(X_samples, predictions, c="red")
    plt.show()


def plot_ols_result(result, X, y):
    prstd, iv_l, iv_u = wls_prediction_std(result)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)

    ax.plot(X, y, "o", label="data")
    ax.plot(X, result.fittedvalues, "r--.", label="OLS")
    ax.plot(X, iv_u, "r--")
    ax.plot(X, iv_l, "r--")
    ax.legend(loc="best")
    plt.show()


def compute_two_stage_least_squares(iv_candidate_df, with_plots=False, verbose=False):
    tf_y_neg_exs = iv_candidate_df["initial X prediction"].values
    tf_y_pos_exs = iv_candidate_df["new X prediction"].values
    acc_y_neg_exs = iv_candidate_df["initial Y prediction"].values
    acc_y_pos_exs = iv_candidate_df["new Y prediction"].values

    tf_X_neg_exs = np.zeros_like(tf_y_neg_exs)
    tf_X_pos_exs = np.ones_like(tf_y_pos_exs)
    np.zeros_like(acc_y_neg_exs)
    np.zeros_like(acc_y_pos_exs)

    tf_X = np.concatenate((tf_X_neg_exs, tf_X_pos_exs))
    tf_y = np.concatenate((tf_y_neg_exs, tf_y_pos_exs))
    acc_y = np.concatenate((acc_y_neg_exs, acc_y_pos_exs))

    stage_1_result = regress_y_on_x(tf_X, tf_y)

    acc_X = stage_1_result.predict(sm.add_constant(tf_X))
    acc_X_pred = pd.Series(acc_X)
    stage_2_result = regress_y_on_x(acc_X_pred, acc_y)

    if with_plots:
        plot_ols_result(stage_1_result, tf_X, tf_y)
        plot_ols_result(stage_2_result, acc_X, acc_y)

    if verbose:
        logging.info("Results of first regression:")
        logging.info(stage_1_result.summary())
        logging.info("************************\n\n\n")
        logging.info("Results of second regression:")
        logging.info(stage_2_result.summary())
        logging.info("************************\n\n\n")
    return stage_2_result


def do_mr_analysis(args):
    iv_candidate_fpath = os.path.join(args.data_dir, args.iv_candidate_fname)
    iv_candidate_df = load_iv_candidates(iv_candidate_fpath)
    two_sls_result = compute_two_stage_least_squares(
        iv_candidate_df, with_plots=args.with_plots, verbose=args.verbose
    )
    logging.info("Causal effects:")
    logging.info(
        "P values: intercept - %1.3f, coefficient - %1.3f",
        *list(two_sls_result.pvalues),
    )
    logging.info(
        "2SLS Params: intercept - %1.3f, coefficient - %1.3f",
        *list(two_sls_result.params)
    )

    logging.info(
        "Confidence interval: intercept - %1.3f, coefficient - %1.3f",
        *two_sls_result.conf_int(alpha=0.05, cols=None),
    )

    wald_ratio, wald_ratio_se, pvalues = compute_wald_ratio(iv_candidate_df)
    logging.info(
        f"Wald ratio: {wald_ratio} (standard error: {wald_ratio_se}) (pvals: {pvalues})"
    )


if __name__ == "__main__":
    logging.getLogger("").setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # Directories / file names
    parser.add_argument(
        "--data_dir",
        default="../dat/",
        help="(Absolute or relative) path to the directory from which we want to pull "
        "data and model files and write output.",
    )
    parser.add_argument(
        "--iv_candidate_fname",
        default="iv_candidates.csv",
        help="Path to .bed file that contains DNase chrom accessibility peaks "
        "(should be relative to `args.data_dir`.",
    )
    parser.add_argument("--with_plots", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    do_mr_analysis(args)

import sys
import pandas as pd
import numpy as np
import scipy.special
from matplotlib import pyplot as plt


def main():
    path = sys.argv[1]
    title = sys.argv[2]
    df = pd.read_csv(path)
    # df2['tf_init'] = scipy.special.logit(df['initial TF prediction'])
    # df2['tf_new'] = scipy.special.logit(df['new TF prediction'])
    # df2['chrom_init'] = scipy.special.logit(df['initial chromatin prediction'])
    # df2['chrom_new'] = scipy.special.logit(df['new chromatin prediction'])
    # df2[~df2.isin([np.nan, np.inf, -np.inf]).any(1)]
    #
    # plt.scatter(df2['tf_init'], df2['chrom_init'],c='b', s=10, alpha=0.3)
    # plt.scatter(df2['tf_new'], df2['chrom_new'],c='r', s=10, alpha=0.3)
    #
    # plt.scatter(df2['tf_init'], df2['chrom_init'], c='b', s=10, alpha=0.3)
    # plt.scatter(df2['tf_new'], df2['chrom_new'], c='r', s=10, alpha=0.3)

    plt.hist(
        df["initial TF prediction"],
        bins=10
    )
    plt.xlabel("Binding Probability")
    plt.ylabel("Number of Predictions")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()

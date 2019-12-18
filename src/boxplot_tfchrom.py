import sys

import numpy as np
import pandas as pd
import scipy.special
from matplotlib import pyplot as plt


def main():
    path = sys.argv[1]
    title = sys.argv[2]
    df = pd.read_csv(path)
    df2 = pd.DataFrame()
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

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    axs[0, 0].boxplot(df['initial TF prediction'].to_numpy())
    axs[0, 0].set_title('initial TF')
    axs[0, 1].boxplot(df['initial chromatin prediction'].to_numpy())
    axs[0, 1].set_title('initial chrom')
    axs[1, 0].boxplot(df['new TF prediction'].to_numpy())
    axs[1, 0].set_title('new TF')
    axs[1, 1].boxplot(df['new chromatin prediction'].to_numpy())
    axs[1, 1].set_title('new chromn')
    axs[0, 0].set_ylim([-0.05, 1.05])
    axs[0, 1].set_ylim([-0.05, 1.05])
    axs[1, 0].set_ylim([-0.05, 1.05])
    axs[1, 1].set_ylim([-0.05, 1.05])
    axs[0, 0].xaxis.set_visible(False)
    axs[0, 1].xaxis.set_visible(False)
    axs[1, 0].xaxis.set_visible(False)
    axs[1, 1].xaxis.set_visible(False)

    plt.show()

if __name__ == '__main__':
    main()

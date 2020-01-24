import sys
import pandas as pd
import numpy as np
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

    plt.scatter(df['initial X prediction'], df['initial Y prediction'], c='b', s=10, alpha=0.8, label='wild type')
    plt.scatter(df['new X prediction'], df['new Y prediction'], c='r', s=10, alpha=0.3, label='in-silico mut')
    plt.xlabel('TF binding prediction')
    plt.ylabel('chromatin accessibility prediction')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.plot([0, 1], [.021,  .021+1.015], color='k', linestyle='-', linewidth=2)
    plt.show()

if __name__ == '__main__':
    main()

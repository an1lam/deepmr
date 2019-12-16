import sys
import pandas as pd
import numpy as np
import scipy.special
from matplotlib import pyplot as plt


def main():
    path = sys.argv[1]
    fig_title = sys.argv[2]
    df = pd.read_csv(path)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(fig_title)
    ax1, ax2 = ax[0, :]
    ax1.scatter(
        df["initial TF prediction"],
        df["new TF prediction"],
        c="b",
        s=10,
        alpha=0.8,
        label="no SNP vs. SNP",
    )
    ax1.set_xlabel("Pre-SNP prediction")
    ax1.set_ylabel("Post-SNP prediction")
    ax1.legend(loc="lower right")
    ax1.set_title("Effect of IV on exposure (prob scale)")
    ax2.scatter(
        df["initial chromatin prediction"],
        df["new chromatin prediction"],
        c="b",
        s=10,
        alpha=0.8,
        label="no SNP vs. SNP",
    )
    ax2.set_xlabel("Pre-SNP prediction")
    ax2.set_ylabel("Post-SNP prediction")
    ax2.set_title("Effect of IV on outcome (prob scale)")
    ax2.legend(loc="lower right")

    ax3, ax4 = ax[1, :]
    ax3.scatter(
        scipy.special.logit(df["initial TF prediction"]),
        scipy.special.logit(df["new TF prediction"]),
        c="b",
        s=10,
        alpha=0.8,
        label="no SNP vs. SNP",
    )
    ax3.set_xlabel("Pre-SNP prediction")
    ax3.set_ylabel("Post-SNP prediction")
    ax3.legend(loc="lower right")
    ax3.set_title("Effect of IV on exposure (log-odds scale)")
    ax4.scatter(
        scipy.special.logit(df["initial chromatin prediction"]),
        scipy.special.logit(df["new chromatin prediction"]),
        c="b",
        s=10,
        alpha=0.8,
        label="no SNP vs. SNP",
    )
    ax4.set_xlabel("Pre-SNP prediction")
    ax4.set_ylabel("Post-SNP prediction")
    ax4.set_title("Effect of IV on outcome (log-odds scale)")
    ax4.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()

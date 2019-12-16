import sys

import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t")

    # print(df.drop(df.columns[5:], axis=1).head())
    df = df.drop(df.columns[5:], axis=1)
    df.to_csv(sys.argv[1], sep="\t", index=False)

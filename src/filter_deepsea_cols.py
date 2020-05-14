import argparse

import pandas as pd

def filter_and_save(args):
     deepsea_features_df = pd.read_csv(args.deepsea_cols_fpath, sep="\t")
     human_tfs_df = pd.read_csv(args.human_tfs_fpath)

     encode_idxs = deepsea_features_df['Data Source'] == 'ENCODE'
     encode_df = deepsea_features_df[encode_idxs]
     hepg2_encode_df = encode_df[encode_df['Cell Type'] == 'HepG2'] 
     original_columns = hepg2_encode_df.columns
     human_tfs_df['TF/DNase/Histone'] = human_tfs_df['HGNC symbol'] 
     hepg2_encode_tfs_df = hepg2_encode_df.merge(human_tfs_df, on='TF/DNase/Histone')
     hepg2_encode_tfs_df[original_columns].to_csv(args.filtered_cols_fpath, index=False)
     
     if args.verbose:
         print(f"Wrote {len(hepg2_encode_tfs_df)} rows to file")
         n_features = pd.unique(hepg2_encode_tfs_df['TF/DNase/Histone'])
         print(f"{len(n_features)} unique features")
         print(f"First few rows\n: {hepg2_encode_tfs_df[original_columns].head()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--deepsea_cols_fpath", default="./deepsea_cols.tsv")
    parser.add_argument("--human_tfs_fpath", default="./human_tfs.csv")
    parser.add_argument("--filtered_cols_fpath", default="./encode_hepg2_deepsea_cols.csv")
    filter_and_save(parser.parse_args())



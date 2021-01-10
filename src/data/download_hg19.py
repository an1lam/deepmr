#!/usr/bin/env python

import argparse
import glob
import logging
import pickle
import os
import subprocess


def download_and_extract_hg19(args):
    dl_target = os.path.join(args.data_dir, "chromFa.tar.gz")
    if not args.force_redownload and os.path.exists(dl_target):
        logging.info("Skipping download and tar process")
        return

    logging.info("Downloading hg19 fasta tar file to '%s'", dl_target)
    dl_url = "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz"
    cmd = "wget %s -O %s" % (dl_url, dl_target)
    subprocess.check_call(cmd, shell=True)

    logging.info("Extracting tar file to '%s'", args.data_dir)
    tar_cmd = "tar -xzvf %s -C %s" % (dl_target, args.data_dir)
    subprocess.check_call(tar_cmd, shell=True)


def glob_chroms_into_hg_file(args):
    hg_path = os.path.join(args.data_dir, "hg19.fa")
    if not args.force_reglob and os.path.exists(hg_path):
        logging.info("Skipping globbing")
        return

    logging.info("Combining chr*.fa files into single hg19 fasta (.fa) file")
    chr_x_path = os.path.join(args.data_dir, "chrX.fa")
    assert os.path.exists(chr_x_path), "Chrom X existence sanity check failed"
    chrom_glob_pattern = os.path.join(args.data_dir, "chr*.fa")
    with open(hg_path, "wb") as f:
        subprocess.check_call("cat %s" % (chrom_glob_pattern), stdout=f, shell=True)

    wc_result = subprocess.check_output(["wc", "-l", hg_path])
    # Assumed format: '<number of lines> <file name>'
    hg_num_lines = int(str(wc_result).split(" ")[0])
    assert (
        hg_num_lines == 62743362
    ), "hg19 line length %d != 62743362, something went wrong..." % (hg_num_lines)

    if not args.cleanup:
        logging.info("Skipping cleanup")
        return

    os.remove("chromFa.tar.gz")
    for chrom_fa in glob.glob(chrom_glob_pattern):
        os.remove(chrom_fa)


def pickle_hg_file(args):
    hg_path = os.path.join(args.data_dir, "hg19.fa")
    logging.info("Pickling '%s'", hg_path)
    hg_pickle_path = os.path.join(args.data_dir, "hg19.pkl")
    hg_chroms = {}
    current_chrom = None
    current_chrom_nts = None
    with open(hg_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if current_chrom is not None:
                    hg_chroms[current_chrom] = "".join(current_chrom_nts)
                current_chrom = line[1:].strip()
                current_chrom_nts = []
            else:
                current_chrom_nts.append(line.upper())

    hg_chroms[current_chrom] = "".join(current_chrom_nts)
    with open(hg_pickle_path, "wb") as pf:
        pickle.dump(hg_chroms, pf)


if __name__ == "__main__":
    logging.getLogger("").setLevel("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../dat/deepsea",
        help="(Absolute or relative) path to the directory in which you want to store data.",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Whether to redownload the HG 19 tar file if it already exists "
        "locally.",
    )
    parser.add_argument(
        "--force_reglob",
        action="store_true",
        help="Whether to combine all the chr*.fa files into hg19.fa if the latter "
        "already exists locally.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Whether to delete tar and unneeded chrom files after globbing everything "
        "into 'hg19.fa'.",
    )
    args = parser.parse_args()
    download_and_extract_hg19(args)
    glob_chroms_into_hg_file(args)
    pickle_hg_file(args)

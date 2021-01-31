import argparse
import subprocess

import pandas as pd


CELL_TYPE = "HepG2"
EXP_FEATURE_COL = "Exposure"


def exp_bed_file_name(exp, sample=False):
    return f"{CELL_TYPE}_{exp}{'_samples' if sample else ''}.gz"


def results_name(exp, out):
    return f"{CELL_TYPE}_{exp}_{out}_mutagenesis_results.csv"


def sample_seqs_cmd(args, input_bed_fname, output_bed_fname):
    # Sample command I ran in the CL:
    #  python pick_random_seqs.py -n 25 \
    #      --input_data_dir "../dat/input/peaks" \
    #      --input_bed_fname "HepG2_FOXA1.gz" \
    #      --output_data_dir "../dat/int/" \
    #      --output_bed_fname "HepG2_FOXA1_samples.bed" \
    #      -c chr3
    cmd = [
        "python",
        "pick_random_seqs.py",
        "-n",
        f"{args.n_seqs}",
        "--input_data_dir",
        f"{args.input_data_dir}",
        "--input_bed_fname",
        f"{input_bed_fname}",
        "--output_data_dir",
        f"{args.intermediate_results_dir}",
        "--output_bed_fname",
        f"{output_bed_fname}",
    ]
    chrom_str = " ".join(chrom for chrom in args.chroms)
    cmd.append(f"-c {chrom_str}")
    return cmd


def in_silico_mutagenesis_cmd(args, exposure, outcome, input_bed_fname, results_fname):
    # Sample command I ran in the CL:
    # python in_silico_mutagenesis.py \
    #   --epochs 50 \
    #   --results_fname effect_sizes__20200430__comparison_new.csv \
    #   --override_random_seed \
    #   --n_seqs 25
    x_column_name = f"{CELL_TYPE}_{exposure}_None"
    y_column_name = f"{CELL_TYPE}_{outcome}_None"
    cmd = [
        "python",
        "in_silico_mutagenesis.py",
        "--epochs",
        f"{args.epochs}",
        "-n",
        f"{args.n_seqs}",
        "--x_column_name",
        x_column_name,
        "--y_column_name",
        y_column_name,
        "--input_data_dir",
        f"{args.intermediate_results_dir}",
        "--peaks_fname",
        f"{input_bed_fname}",
        "--output_data_dir",
        f"{args.results_dir}",
        "--preds_fname",
        f"{results_fname.replace('.csv', '.pickle')}",
        "--results_fname",
        f"{results_fname}",
    ]
    if args.override_random_seed:
        cmd.append("--override_random_seed")
    return cmd


def run(args):
    features_df = pd.read_csv(args.features_fpath)
    features_df = features_df.drop_duplicates(subset=[EXP_FEATURE_COL])

    for row in exps_df.iterrows():
        current_exp = row[1][EXP_FEATURE_COL]
        current_out = row[1][OUT_FEATURE_COL]
        if args.limit_to_exps is not None and current_exp not in args.limit_to_exps:
            continue

        input_bed_fname = exp_bed_file_name(current_exp)
        output_bed_fname = exp_bed_file_name(current_exp, sample=(not args.skip_sampling))

        cmd = sample_seqs_cmd(args, input_bed_fname, output_bed_fname)
        if not args.skip_sampling:
            if not args.dry_run:
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as cpe:
                    print(cpe)
                    raise cpe
            else:
                print(f"Pick random seqs command: {cmd}")

        mutagenesis_results_fname = results_name(current_exp, current_out)
        cmd = in_silico_mutagenesis_cmd(
            args, current_exp, current_out, output_bed_fname, mutagenesis_results_fname
        )
        if not args.skip_mutagenesis:
            if not args.dry_run:
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as cpe:
                    print(cpe)
                    raise cpe
            else:
                print(f"Mutagenesis command: {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--skip_sampling", action="store_true", default=False)
    parser.add_argument("--skip_mutagenesis", action="store_true", default=False)
    parser.add_argument("--input_data_dir", default="../dat/deepsea/")
    parser.add_argument("--intermediate_results_dir", default="../dat/int/")
    parser.add_argument(
        "-c",
        "--chroms",
        nargs="+",
        metavar="N",
        default=["chr2", "chr3"],
        help="Chromosomes to use as a validation set. Typically have format "
        "'chr<number | letter (X or Y)>'.",
    )
    parser.add_argument("--results_dir", default="../dat/deepsea/res/")

    parser.add_argument("--features_fpath", default="./encode_hepg2_deepsea_cols.csv")
    parser.add_argument("-l", "--limit_to_exps", nargs="+")
    parser.add_argument("--n_seqs", type=int, default=25)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--y_column_feature", default="DNase")
    parser.add_argument("--override_random_seed", action="store_true")

    run(parser.parse_args())

"""
Simulate transcription factor binding cooperativity data.
"""
import argparse
from collections import OrderedDict
import logging
import os

import editdistance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import simdna
from simdna import synthetic

from np_utils import convolve_1d
from pyx.one_hot import one_hot


def add_args(parser):
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "-l",
        "--sequence_length",
        type=int,
        help="Length of simulated DNA sequences",
        default=100,
    )
    parser.add_argument(
        "--exposure_motif",
        default="GATA_disc1",
        help="Name of motif for the cause TF in the simulation",
    )
    parser.add_argument(
        "--outcome_motif",
        default="TAL1_known1",
        help="Name of motif for the effect TF in the simulation",
    )
    parser.add_argument(
        "--train_sequences",
        type=int,
        default=50000,
        help="Number of sequences to generate for NN training set",
    )
    parser.add_argument(
        "--test_sequences",
        type=int,
        default=50000,
        help="Number of sequences to generate for NN test set",
    )
    parser.add_argument(
        "--variant_augmentation_percentage",
        type=float,
        default=0.0,
        help="Fraction of train/test sequences to duplicate and mutate.",
    )

    parser.add_argument(
        "--data_dir",
        default="../dat/sim/",
        help="Path to directory from/to which to read/write data",
    )
    parser.add_argument(
        "--train_data_fname",
        default="train_labels.csv",
        help="Name of the file to which training sequences and labels will be saved",
    )
    parser.add_argument(
        "--test_data_fname",
        default="test_labels.csv",
        help="Name of the file to which test sequences and labels will be saved",
    )
    parser.add_argument(
        "--train_variant_data_fname",
        default="train_variant_labels.csv",
        help="Name of the file to which training sequences and labels will be saved",
    )
    parser.add_argument(
        "--test_variant_data_fname",
        default="test_variant_labels.csv",
        help="Name of the file to which test sequences and labels will be saved",
    )
    parser.add_argument(
        "--log_summary_stats",
        action="store_true",
        help="Whether to log summary statistics about the counts at the end of the simulation.",
    )

    return parser


def ddg_pwm_score(one_hot_sequences, pwm, mu=0):
    """
    Score one-hot encoded sequences using a PWM.

    The algorithm used closely follows the one Zhao & Stormo describe here:
        https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.1893/MediaObjects/41587_2011_BFnbt1893_MOESM84_ESM.pdf

    Briefly, we:
    1. Convert the PWM to a negative log-likelihood ratio relative to the
       background frequency.
    2. Convolve it over each one-hot encoded sequence (along the base axis).
    3. "Invert" the scores back to non-log space via a quasi-sigmoid.

    Regarding 3, what we"re doing is only a quasi-sigmoid because technically
    we"re not applying to a true logit (log-odds ratio), just a ratio of two
    two probabilities that wouldn"t sum to 1. This is because Zhao & Stormo
    are treating the NLL ratio as an energy rather than a true likelihood ratio.
    """
    assert ((pwm > 0) & (pwm < 1)).all()
    background_pwm = np.array([background_frequency[nt] for nt in "ACTG"])
    nll_pwm = -np.log(pwm / background_pwm)
    nll_scores = convolve_1d(one_hot_sequences, nll_pwm.T)
    # Convert back to probability space
    return 1 / (1 + np.exp(nll_scores - mu))


def simulate_counts(sequences, exposure_pwm, outcome_pwm):
    """
    Assign each sequence a "count" for exposure / outcome TFs by summing the PWM scores for each.

    In more detail, to compute counts we take the following steps:
    1. Compute each PWM to a negative log-likelihood ratio relative to the background frequency.
    2. Slide the two PWMs over each sequence and sum the scores over the length of the sequence
       (result: q_{t, exp}, q_{t, out}).
    """

    one_hot_sequences = np.array([one_hot(sequence) for sequence in sequences])
    # Convert to negative log-likelihood ratio for numerical stability.
    q_exp = 1 - np.prod(1 - ddg_pwm_score(one_hot_sequences, exposure_pwm), axis=-1)
    q_out = 1 - np.prod(1 - ddg_pwm_score(one_hot_sequences, outcome_pwm), axis=-1)
    return q_exp, q_out


def simulate_oracle_predictions(
    sequences,
    exposure_pwm,
    outcome_pwm,
    alpha=100,
    beta=100,
    exp_bias=0,
    out_bias=0,
):
    """
    Simulate oracle predictions given counts from a simulated assay.

    Compute exposure and outcome counts as follows:
        c_{t, exp} = alpha * q_{t, exp}
        c_{t, out} = beta * (q_{t, exp} * q_{t, out}).

       The design of these formulas is driven by the goal of having the exposure be a linear function
       of the PWM score and some scalar and the output be a multiplicative function of the exposure
       and outcome scores and some scalar.
    """
    q_exp, q_out = simulate_counts(sequences, exposure_pwm, outcome_pwm)
    c_exp = exp_bias + alpha * q_exp
    c_out = out_bias + beta * (q_exp * q_out)

    c_exp_noisy = np.random.poisson(c_exp, len(c_exp))
    c_out_noisy = np.random.poisson(c_out, len(c_out))
    return c_exp_noisy, c_out_noisy, c_exp, c_out, q_exp, q_out


nts_to_ints = {"A": 0, "C": 1, "G": 2, "T": 3}
ints_to_nts = {i: nt for nt, i in nts_to_ints.items()}


def mutate_nt(nt):
    return ints_to_nts[(nts_to_ints[nt] + np.random.choice([1, 2, 3])) % 4]


def mutate_sequence(sequence, embedding):
    if embedding is not None:
        start_pos = embedding.startPos
        embedded_seq = embedding.what.string
        mutation_pos = start_pos + np.random.choice(np.arange(len(embedded_seq)))
    else:
        mutation_pos = np.random.choice(np.arange(len(sequence)))

    new_nt = mutate_nt(sequence[mutation_pos])
    new_sequence = sequence[:mutation_pos] + new_nt + sequence[mutation_pos + 1 :]
    return new_sequence


def mutate_sequences(sequences, embeddings, max_mutations_per_variant=1):
    variants = []
    for sequence, embeddings_ in zip(sequences, embeddings):
        if len(embeddings_) == 1:
            variant = mutate_sequence(sequence, embeddings_[0])
        elif len(embeddings_) == 3:
            variant = mutate_sequence(
                mutate_sequence(sequence, embeddings_[0]), embeddings_[1]
            )
        else:
            assert len(embeddings_) == 0
            variant = mutate_sequence(sequence, None)
        variants.append(variant)
        
    return variants


def generate_variant_counts_and_labels(
    sequences,
    labels,
    embeddings,
    exposure_pwm,
    outcome_pwm,
    frac=1.0,
    max_mutations_per_variant=1,
):
    n_variants = int(len(sequences) * frac)
    variant_indexes = np.random.choice(np.arange(len(sequences)), size=n_variants).astype(np.int)
    variants = [sequences[i] for i in variant_indexes].copy()
    variants = mutate_sequences(
        variants, embeddings, max_mutations_per_variant=max_mutations_per_variant
    )
    variant_counts = simulate_oracle_predictions([str(v) for v in variants], exposure_pwm, outcome_pwm)
    variant_labels = labels[variant_indexes].copy()
    return variants, variant_counts, variant_labels, variant_indexes


background_frequency = OrderedDict([("A", 0.27), ("C", 0.23), ("G", 0.23), ("T", 0.27)])


def generate_sequences_mixture(
    motifs,
    exposure_motif,
    outcome_motif,
    n_train_sequences=100,
    n_test_sequences=10,
    sequence_length=100,
):
    # Setup generator for background nucleotide distribution
    background_generator = synthetic.ZeroOrderBackgroundGenerator(
        seqLength=sequence_length,
        discreteDistribution=background_frequency,
    )

    # Set up embedders for the two motif"s PWMs
    position_generator = synthetic.UniformPositionGenerator()
    motif_lengths = [
        len(motifs.getPwm(name).getRows()) for name in [exposure_motif, outcome_motif]
    ]

    # Generate sequences
    spacing_generator = synthetic.UniformIntegerGenerator(
        max(motif_lengths), sequence_length - sum(motif_lengths)
    )
    exposure_motif_generator = synthetic.PwmSamplerFromLoadedMotifs(
        motifs, exposure_motif
    )
    outcome_motif_generator = synthetic.PwmSamplerFromLoadedMotifs(
        motifs, outcome_motif
    )
    embedders = [
        synthetic.SubstringEmbedder(
            substringGenerator=exposure_motif_generator,
            positionGenerator=position_generator,
            name=exposure_motif,
        ),
        synthetic.SubstringEmbedder(
            substringGenerator=outcome_motif_generator,
            positionGenerator=position_generator,
            name=outcome_motif,
        ),
        synthetic.EmbeddableEmbedder(
            synthetic.PairEmbeddableGenerator(
                embeddableGenerator1=exposure_motif_generator,
                embeddableGenerator2=outcome_motif_generator,
                separationGenerator=spacing_generator,
            )
        ),
    ]
    overall_embedder = synthetic.RandomSubsetOfEmbedders(
        synthetic.BernoulliQuantityGenerator(0.75), embedders
    )
    sequence_sim = synthetic.EmbedInABackground(
        backgroundGenerator=background_generator, embedders=[overall_embedder]
    )
    train_sequences = synthetic.GenerateSequenceNTimes(sequence_sim, n_train_sequences)
    test_sequences = synthetic.GenerateSequenceNTimes(sequence_sim, n_test_sequences)
    return train_sequences, test_sequences


def main(args):
    os.makedirs(args.data_dir, exist_ok=True)
    if args.seed is not None:
        np.random.seed(seed=args.seed)

    # Load motif repository
    motifs = synthetic.LoadedEncodeMotifs(
        simdna.ENCODE_MOTIFS_PATH, pseudocountProb=0.00001
    )

    train_sequences, test_sequences = generate_sequences_mixture(
        motifs,
        args.exposure_motif,
        args.outcome_motif,
        n_train_sequences=args.train_sequences,
        n_test_sequences=args.test_sequences,
        sequence_length=args.sequence_length,
    )
    # Save raw sequences to labeled files
    def assign_labels(self, generated_sequence):
        has_exposure_motif = generated_sequence.additionalInfo.isInTrace(
            args.exposure_motif
        )
        has_outcome_motif = generated_sequence.additionalInfo.isInTrace(
            args.outcome_motif
        )
        has_both_motifs = generated_sequence.additionalInfo.isInTrace(
            "EmbeddableEmbedder"
        )
        return [
            int(has_exposure_motif or has_both_motifs),
            int(has_outcome_motif or has_both_motifs),
            int(has_both_motifs),
        ]

    label_names = ["has_exposure", "has_outcome", "has_both"]
    synthetic.printSequences(
        os.path.join(args.data_dir, "train_sequences.simdata"),
        train_sequences,
        includeFasta=True,
        includeEmbeddings=True,
        labelGenerator=synthetic.LabelGenerator(
            labelNames=label_names,
            labelsFromGeneratedSequenceFunction=assign_labels,
        ),
        prefix="train",
    )
    synthetic.printSequences(
        os.path.join(args.data_dir, "test_sequences.simdata"),
        test_sequences,
        includeFasta=True,
        includeEmbeddings=True,
        labelGenerator=synthetic.LabelGenerator(
            labelNames=label_names,
            labelsFromGeneratedSequenceFunction=assign_labels,
        ),
        prefix="test",
    )

    # Fetch training & test data
    train_sim_data = synthetic.read_simdata_file(
        os.path.join(args.data_dir, "train_sequences.simdata")
    )
    test_sim_data = synthetic.read_simdata_file(
        os.path.join(args.data_dir, "test_sequences.simdata")
    )
    train_sequences = train_sim_data.sequences
    test_sequences = test_sim_data.sequences
    train_labels = train_sim_data.labels
    test_labels = test_sim_data.labels
    train_embeddings = train_sim_data.embeddings
    test_embeddings = test_sim_data.embeddings

    # Generate count labels for labels
    exposure_pwm = motifs.loadedMotifs[args.exposure_motif].getRows()
    outcome_pwm = motifs.loadedMotifs[args.outcome_motif].getRows()
    train_counts = simulate_oracle_predictions(
        train_sequences, exposure_pwm, outcome_pwm
    )
    test_counts = simulate_oracle_predictions(test_sequences, exposure_pwm, outcome_pwm)

    if args.log_summary_stats:
        fig, axs = plt.subplots(2, 1, figsize=(6, 10))
        axs[0].scatter(train_counts[4], train_counts[2], alpha=0.3)
        axs[0].set_xlabel("$ q_{\\rm{exp}} $")
        axs[0].set_ylabel("Expected Counts")

        axs[1].scatter(train_counts[5], train_counts[5])
        axs[1].set_xlabel("$ q_{\\rm{out}} $")
        axs[1].set_ylabel("Expected Counts")
        plt.savefig(os.path.join(args.data_dir, "counts_vs_q_plot.png"))

    train_df = pd.DataFrame(
        {
            "sequences": train_sequences,
            "labels_exp": train_counts[0],
            "labels_out": train_counts[1],
            "has_exposure": train_labels[:, 0],
            "has_outcome": train_labels[:, 1],
            "has_both": train_labels[:, 2],
        }
    )
    test_df = pd.DataFrame(
        {
            "sequences": test_sequences,
            "labels_exp": test_counts[0],
            "labels_out": test_counts[1],
            "has_exposure": test_labels[:, 0],
            "has_outcome": test_labels[:, 1],
            "has_both": test_labels[:, 2],
        }
    )
    # Save labeled data to file to be used for model training and predictions
    train_df.to_csv(
        os.path.join(args.data_dir, args.train_data_fname), header=True, index=False
    )
    test_df.to_csv(
        os.path.join(args.data_dir, args.test_data_fname), header=True, index=False
    )

    if args.variant_augmentation_percentage > 0:

        (
            train_variants,
            train_variant_counts,
            train_variant_labels,
            train_variant_indexes,
        ) = generate_variant_counts_and_labels(
            train_sequences,
            train_labels,
            train_embeddings,
            exposure_pwm,
            outcome_pwm,
            frac=args.variant_augmentation_percentage,
        )
        (
            test_variants,
            test_variant_counts,
            test_variant_labels,
            test_variant_indexes,
        ) = generate_variant_counts_and_labels(
            test_sequences,
            test_labels,
            test_embeddings,
            exposure_pwm,
            outcome_pwm,
            frac=args.variant_augmentation_percentage,
        )
        train_variant_df = pd.DataFrame(
            {
                "sequences": train_variants,
                "labels_exp": train_variant_counts[0],
                "labels_out": train_variant_counts[1],
                "has_exposure": train_variant_labels[:, 0],
                "has_outcome": train_variant_labels[:, 1],
                "has_both": train_variant_labels[:, 2],
                "original_index": train_variant_indexes,
            }
        )
        test_variant_df = pd.DataFrame(
            {
                "sequences": test_variants,
                "labels_exp": test_variant_counts[0],
                "labels_out": test_variant_counts[1],
                "has_exposure": test_variant_labels[:, 0],
                "has_outcome": test_variant_labels[:, 1],
                "has_both": test_variant_labels[:, 2],
                "original_index": test_variant_indexes,
            }
        )
        train_variant_df.to_csv(
            os.path.join(args.data_dir, args.train_variant_data_fname), header=True, index=False
        )
        test_variant_df.to_csv(
            os.path.join(args.data_dir, args.test_variant_data_fname), header=True, index=False
        )


    if args.log_summary_stats:
        logging.info(
            "Training count summary stats: "
            + "\n\texposure mean = %.2f, variance = %.2f \n\toutcome mean = %.2f, variance = %.2f"
            % (
                train_df.labels_exp.mean(),
                train_df.labels_exp.var(),
                train_df.labels_out.mean(),
                train_df.labels_out.var(),
            )
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    main(parser.parse_args())

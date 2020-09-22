"""
Simulate transcription factor binding cooperativity data.
"""
import argparse
from collections import OrderedDict

import numpy as np
import simdna
from simdna import synthetic

from np_utils import convolve_1d
from pyx.one_hot import one_hot


def add_args(parser):
    parser.add_argument(
        "-n",
        "--n_sequences",
        type=int,
        help="Number of sequences to simulate",
        default=1000,
    )
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
        "-s",
        "--save_raw_sequences",
        action="store_true",
        help="Whether to save the sequences strings as-is after generating them",
    )

    return parser


def simulate_counts(sequences, exposure_pwm, outcome_pwm, alpha=1, beta=1):
    """
    Assign each sequence a "count" for exposure / outcome TFs based on the simulated causal relationship.

    In more detail, to compute counts we take the following steps:
    1. Compute each PWM to a negative log-likelihood ratio relative to the background frequency.
    2. Slide the two PWMs over each sequence and sum the scores over the length of the sequence
       (result: q_{t, exp}, q_{t, out}).
    3. Compute exposure and outcome counts as follows:
        c_{t, exp} = alpha * q_{t, exp}
        c_{t, out} = beta * (q_{t, exp} * q_{t, out}).
       The design of these formulas is driven by the goal of having the exposure be a linear function
       of the PWM score and some scalar and the output be a multiplicative function of the exposure
       and outcome scores and some scalar.
    """

    background_pwm = np.array([background_frequency[nt] for nt in "ACTG"])
    exposure_nll_pwm = -np.log(exposure_pwm / background_pwm)
    outcome_nll_pwm = -np.log(outcome_pwm / background_pwm)
    one_hot_sequences = np.array([one_hot(sequence) for sequence in sequences])
    q_exp = np.sum(convolve_1d(one_hot_sequences, exposure_nll_pwm.T), axis=-1)
    q_out = np.sum(convolve_1d(one_hot_sequences, outcome_nll_pwm.T), axis=-1)
    return q_exp * alpha, q_out * q_exp * beta


background_frequency = OrderedDict([("A", 0.27), ("C", 0.23), ("G", 0.23), ("T", 0.27)])


def main(args):
    # Setup generator for background nucleotide distribution
    background_generator = synthetic.ZeroOrderBackgroundGenerator(
        seqLength=args.sequence_length,
        discreteDistribution=background_frequency,
    )

    # Load motif repository
    motifs = synthetic.LoadedEncodeMotifs(
        simdna.ENCODE_MOTIFS_PATH, pseudocountProb=0.001
    )

    # Set up embedders for the two motif's PWMs
    position_generator = synthetic.UniformPositionGenerator()
    pwm_samplers = [
        (
            args.exposure_motif,
            synthetic.PwmSamplerFromLoadedMotifs(motifs, args.exposure_motif),
        ),
        (
            args.outcome_motif,
            synthetic.PwmSamplerFromLoadedMotifs(motifs, args.outcome_motif),
        ),
    ]
    pwm_embedders = [
        synthetic.SubstringEmbedder(
            substringGenerator=sampler,
            positionGenerator=position_generator,
            name=name,
        )
        for name, sampler in pwm_samplers
    ]
    # Want a 4-way split between neither motif, motif 1, motif 2, and both motifs.
    # In order to achieve this, we construct two embedders, one for each motif.
    # Each embedder should include its underlying with probability 1/2.
    # By basically probability rules, the probability of two fair independent binary
    # random variables both equaling 1 or 0 equals 1/4.
    pwm_or_nothing_embedders = [
        synthetic.XOREmbedder(
            embedder1=pwm_embedder,
            # Use random subset embedder to construct null embedder.
            embedder2=synthetic.RandomSubsetOfEmbedders(
                quantityGenerator=synthetic.FixedQuantityGenerator(0), embedders=[]
            ),
            probOfFirst=0.5,
        )
        for pwm_embedder in pwm_embedders
    ]

    # Generate sequences
    sequence_sim = synthetic.EmbedInABackground(
        backgroundGenerator=background_generator, embedders=pwm_or_nothing_embedders
    )
    train_sequences = synthetic.GenerateSequenceNTimes(
        sequence_sim, args.train_sequences
    )
    test_sequences = synthetic.GenerateSequenceNTimes(sequence_sim, args.test_sequences)

    # (Optional:) Save raw sequences to labeled files for checking
    if args.save_raw_sequences:

        def assign_labels(self, generated_sequence):
            has_exposure_motif = generated_sequence.additionalInfo.isInTrace(
                args.exposure_motif
            )
            has_outcome_motif = generated_sequence.additionalInfo.isInTrace(
                args.outcome_motif
            )
            has_both_motifs = has_exposure_motif and has_exposure_motif
            return [
                int(has_exposure_motif),
                int(has_outcome_motif),
                int(has_both_motifs),
            ]

        label_names = ["has_exposure", "has_outcome", "has_both"]
        synthetic.printSequences(
            "train_sequences.simdata",
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
            "test_sequences.simdata",
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
    train_sim_data = synthetic.read_simdata_file("train_sequences.simdata")
    test_sim_data = synthetic.read_simdata_file("test_sequences.simdata")
    train_sequences = train_sim_data.sequences
    test_sequences = test_sim_data.sequences

    # Generate count labels for labels
    exposure_pwm = motifs.loadedMotifs[args.exposure_motif].getRows()
    outcome_pwm = motifs.loadedMotifs[args.outcome_motif].getRows()
    train_counts = simulate_counts(train_sequences, exposure_pwm, outcome_pwm)
    test_counts = simulate_counts(train_sequences, exposure_pwm, outcome_pwm)
    print(train_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    main(parser.parse_args())

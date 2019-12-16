import argparse
import logging
import timeit
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from data_loader import BedPeaksDataset
from data_loader import load_accessibility_data
from data_loader import load_genome
from models import CNN_1d
from models import get_default_cnn
from models import get_big_cnn
from train import test_model
from utils import detect_device
from utils import load_model


def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):
    torch.set_grad_enabled(train_flag)
    cnn_1d.train() if train_flag else cnn_1d.eval()

    losses = []
    accuracies = []

    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device))  # transfer data to GPU

        output = cnn_1d(x)  # forward pass
        output = output.squeeze()  # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits(output, y)  # numerically stable

        if train_flag:
            loss.backward()  # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float())
        accuracies.append(accuracy.detach().cpu().numpy())

    return (np.mean(losses), np.mean(accuracies))


def create_and_train_model(
    train_dataset,
    val_dataset,
    cnn_1d=CNN_1d(),
    epochs=100,
    patience=15,
    check_point_filename="cnn_1d_checkpoint.pt",
):
    """
    Instantiate, train, and return a 1D CNN model and its accuracy metrics.
    """
    # Reload data
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1000, num_workers=0
    )
    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1000, num_workers=0
    )

    # Set up model and optimizer
    device = detect_device()
    cnn_1d.to(device)
    optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True)

    # Training loop w/ early stopping
    train_accs = []
    val_accs = []
    patience_counter = patience
    best_val_loss = np.inf
    # to save the best model fit to date
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(
            True, train_dataloader, cnn_1d, optimizer, device
        )
        val_loss, val_acc = run_one_epoch(
            False, validation_dataloader, cnn_1d, optimizer, device
        )
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            torch.save(cnn_1d.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                # recover the best model so far
                cnn_1d.load_state_dict(torch.load(check_point_filename))
                break

        elapsed = float(timeit.default_timer() - start_time)
        log_line = """
            Epoch %i took %.2fs. Train loss: %.4f acc: %.4f.  Val loss: %.4f
            acc: %.4f. Patience left: %i
        """
        print(
            log_line
            % (
                epoch + 1,
                elapsed,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                patience_counter,
            )
        )
    return cnn_1d, train_accs, val_accs


def main(args):
    logging.getLogger("").setLevel("INFO")
    torch.manual_seed(2)
    model = get_big_cnn()
    genome = load_genome(args.data_dir, genome_file="hg19.pkl")
    if args.mode == "train":
        train_chroms = list(set(genome.keys()) - set(args.val_chroms + args.test_chroms))
        train_data = load_accessibility_data(
            args.data_dir, train_chroms, args.bed_file_path, verbose=True
        )
        validation_data = load_accessibility_data(
            args.data_dir, args.val_chroms, args.bed_file_path, verbose=True
        )
        train_dataset = BedPeaksDataset(train_data, genome, model.seq_len)
        validation_dataset = BedPeaksDataset(validation_data, genome, model.seq_len)
        model_file_path = os.path.join(args.data_dir, args.model_file_path)
        model, _, _ = create_and_train_model(
            train_dataset,
            validation_dataset,
            cnn_1d=model,
            check_point_filename=model_file_path,
        )
        torch.save(model.state_dict(), os.path.join(args.data_dir, args.model_file_path))

    elif args.mode == "test":
        model = load_model(args, args.model_file_path, model)
        test_chroms = args.test_chroms
        test_data = load_accessibility_data(args.data_dir, args.test_chroms, args.accessibility_file, verbose=True)
        print(test_data.head())
        print(len(test_data))
        test_dataset = BedPeaksDataset(test_data, genome, model.seq_len)
        model_file_path = os.path.join(args.data_dir, args.model_file_path)
        model, test_accs = test_model(test_dataset, model)
        print(f"Test accuracy: {test_accs}")

if __name__ == "__main__":
    logging.getLogger("").setLevel("INFO")
    parser = argparse.ArgumentParser()
    # Directories / file names
    parser.add_argument(
        "--data_dir",
        default="../dat/",
        help="(Absolute or relative) path to the directory from which we want to pull "
        "data and model files and write output.",
    )
    parser.add_argument(
        "--bed_file_path",
        default="DNASE.A549.conservative.narrowPeak.gz",
        help="Path to .bed file that contains DNase chrom accessibility peaks "
        "(should be relative to `args.data_dir`.",
    )
    parser.add_argument(
        "--model_file_path",
        default="chrom_acc_cnn_1d.pt",
        help="Path to saved version of TF binding"
        "prediction model. Path will be appended to '--data_dir' arg.",
    )

    # Configuration
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "-c",
        "--val_chroms",
        nargs="+",
        default=["chr1", "chr2", "chr3"],
        help="Chromosomes to use as a validation set. Typically have format "
        "'chr<number | letter (X or Y)>'.",
    )
    parser.add_argument(
        "--test_chroms",
        nargs="+",
        default=["chr1"],
        help="Chromosomes to hold out as a test set. Typically have format "
        "'chr<number | letter (X or Y)>'.",
    )
    parser.add_argument("--accessibility_file")
    args = parser.parse_args()
    main(args)

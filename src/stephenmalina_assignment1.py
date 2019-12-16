# -*- coding: utf-8 -*-
import math
import pickle
import timeit

import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from data_loader import BedPeaksDataset
from pyx.one_hot import one_hot

# Pytorch basics

# `pytorch` (as opposed to e.g. Theano, Tensorflow 1) uses *eager execution*: this lets you write computations as python code that you can test and debug, and later
# 1.   Backprop through (i.e. get gradients with respect to inputs)
# 2.   Run on the GPU for (hopefully!) big speedups.
#
# Here's an example:

if __name__ == "__main__":
    x_np = one_hot("CCGCGNGGNGGCAG")
    x_tensor = torch.tensor(x_np)
    print(x_tensor)
    torch.sum(x_tensor, 1)
    torch.manual_seed(2)  # I played with different initialization here!

    """
    Loading data

    To test this out we need some data! Our task will be predict binding of the important transcriptional repressor [CTCF](https://en.wikipedia.org/wiki/CTCF) in a human lung cancer cell line called [A549](https://en.wikipedia.org/wiki/A549_cell). The data is available from ENCODE [here](https://www.encodeproject.org/experiments/ENCSR035OXA/) but we processed it a bit to a
    1. Merge replicate experiments using the "irreproducible discovery rate" (IDR) [paper](https://arxiv.org/abs/1110.4705) [code](https://github.com/spundhir/idr)
    2. Remove chromosome 1 as test data for a [kaggle InClass](https://www.kaggle.com/c/predict-ctcf-binding) (more details further down!)

    Genomics data representing discrete binding events is typically stored in the  `bed` format, which is described on the UCSC Genome Browser [website](https://genome.ucsc.edu/FAQ/FAQformat.html#format1).
    """

    DATADIR = "../dat/"
    binding_data = pd.read_csv(
        DATADIR + "CTCF_A549_train.bed.gz",
        sep="\t",
        names=("chrom", "start", "end", "name", "score"),
    )
    binding_data[10:15]

    """`start` and `end` are positions in the genome. `name` isn't used here. `score` is the `-log10(pvalue)` for a test of whether there is significant binding above background. We'll ignore this for now (since all these peaks are statistically significant), but you could try to incorporate it in your model if you want.

    We'll split the binding data into training and validation just by taking out chromosomes 2 and 3, which represents about 15% of the training data we have:
    """

    validation_chromosomes = ["chr2", "chr3"]
    train_data = binding_data[~binding_data["chrom"].isin(validation_chromosomes)]
    validation_data = binding_data[binding_data["chrom"].isin(validation_chromosomes)]
    print(len(validation_data) / len(binding_data))

    """We'll also need the human genome, which we provide here as a pickle since it's fast(ish) to load compared to reading in a txt file.

    It's worth knowing that the human genome has different *versions* that are released as more missing parts are resolved by continued sequencing and assembly efforts. Version `GRCh37` (also called `hg19`) was released in 2009, and `GRCh38` (`hg38`) was released in 2013. We'll be using `hg38`, but lots of software and data still use `hg19` (!) though so always check.
    """

    genome = pickle.load(open(DATADIR + "hg38.pkl", "rb"))

    # `genome` is just a dictionary where the keys are the chromosome
    # names and the values are strings representing the actual DNA:
    print("Example genome snippet:", genome["chr13"][100000000:100000010])

    # A substantial proportion of each chromosome is "N"s:"""
    print(
        "Fraction of genome made up of missingness:",
        genome["chr13"].count("N") / len(genome["chr13"]),
    )

    # Ns represents "missing" regions, typically because the region has
    # too many repetitive sequences making mapping impossible, which is
    # especially the case in
    # [centrosomes](https://en.wikipedia.org/wiki/Centrosome) and
    # [telomeres](https://en.wikipedia.org/wiki/Telomere).
    # Resolving these difficult to map regions is an ongoing effort.
    #
    # We'll use the `torch` data loading utilities (nicely documented
    # [here](https://pytorch.org/docs/stable/data.html)) to handle
    # 1. Grouping individual (x,y) pairs into minibatches.
    # 2. Converting `numpy` arrays into `torch` tensors.
    #
    # We could also use `num_workers>0` to have a background process generating
    # the next batch using the CPU while the GPU is working, but in my
    # experience this actually slows things down. If you were in a computer
    # vision setting where you wanted to do intensive [data
    # augmentation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    # it might make a difference.

    train_dataset = BedPeaksDataset(train_data, genome, 100)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers=0)

    # Training
    #
    # Let's train our super simple CNN using stochastic gradient descent:
    #
    #
    # start_time = timeit.default_timer()
    #
    # torch.set_grad_enabled(True)  # we'll need gradients
    #
    # for epoch in range(20):  # run for 20 epochs
    #     losses = []
    #     accuracies = []
    #     for (x, y) in train_dataloader:  # iterate over minibatches
    #
    #         output = my_simplest_CNN(x)  # forward pass
    #         # in practice (and below) we'll use more numerically stable built-in
    #         # functions for the loss
    #         loss = - torch.mean(y * torch.log(output) +
    #                             (1.-y) * torch.log(1.-output))
    #         loss.backward()  # back propagation
    #
    #         # iterate over parameter tensors: just the layer1 weights and bias here
    #         for parameters in my_first_conv_layer.parameters():
    #             # in practive reduce or adapt learning rate
    #             parameters.data -= 1.0 * parameters.grad
    #             parameters.grad.data.zero_()  # torch accumulates gradients so need to reset
    #
    #         losses.append(loss.detach().numpy())  # convert back to numpy
    #         accuracy = torch.mean(((output > .5) == (y > .5)).float())
    #         accuracies.append(accuracy.detach().numpy())
    #
    #     elapsed = float(timeit.default_timer() - start_time)
    #     print("Epoch %i %.2fs/epoch Loss: %.4f Acc: %.4f" %
    #           (epoch+1, elapsed/(epoch+1), np.mean(losses), np.mean(accuracies)))
    #
    # """So we can get pretty decent accuracy even with this very simple CNN (although I cheated a bit by trying a few random seeds and knowing that the CTCF consensus motif is 14nt long so would require a width 14 filter)."""
    #
    # validation_dataset = BedPeaksDataset(validation_data, genome, 100)
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=1000)
    # accuracies = [torch.mean(((my_simplest_CNN(x) > .5) == (y > .5)).float(
    # )).detach().cpu().numpy() for (x, y) in validation_dataloader]
    # np.mean(accuracies)
    #
    # """The validation accuracy is very similiar to the train accuracy, suggesting we're not overfitting (as we would expect with such a simple model).
    #
    #
    # """The `torch` output nicely illustrates the various layers in our network. We have two convolutional units each doing conv -> batchnorm -> maxpooling -> activation, followed by two layers of dense network, including a dropout layer.
    #
    # We'll recreate the dataloaders to satisfy the input sequence length requirement of the new model.
    # """
    #
    # train_dataset = BedPeaksDataset(train_data, genome, cnn_1d.seq_len)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1000, num_workers=0)
    # validation_dataset = BedPeaksDataset(validation_data, genome, cnn_1d.seq_len)
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=1000)
    #
    # """## Transfering to the GPU
    # The previous example actually run on the CPU. Now we're using a more complex model it'll be worth running on the GPU. To do that we need to remember to transfer both model and data. Transfering the model:
    # """
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cnn_1d.to(device)
    # print(device)
    #
    # """Check that this says `cuda` not `cpu`! You need to go to Runtime -> Change runtime type -> Set "Hardware Accelerator" -> GPU (or TPU) if not.
    #
    # Next we'll set up the optimizer. `torch` has a number of [optimizers](https://pytorch.org/docs/stable/optim.html), all of which are variants of SGD. [Adam](https://arxiv.org/abs/1412.6980) with the [AMSgrad convergence fix](https://openreview.net/forum?id=ryQu7f-RZ) is a good default in my experience.
    # """
    #
    # optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True)
    #
    # """We define a training loop which can be used for both training and validation loops by setting the `train_flag`."""
    #
    #
    # def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):
    #
    #     torch.set_grad_enabled(train_flag)
    #     cnn_1d.train() if train_flag else cnn_1d.eval()
    #
    #     losses = []
    #     accuracies = []
    #
    #     for (x, y) in dataloader:
    #
    #         (x, y) = (x.to(device), y.to(device))  # transfer data to GPU
    #
    #         output = cnn_1d(x)  # forward pass
    #         output = output.squeeze()  # remove spurious channel dimension
    #         loss = F.binary_cross_entropy_with_logits(
    #             output, y)  # numerically stable
    #
    #         if train_flag:
    #             loss.backward()  # back propagation
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #         losses.append(loss.detach().cpu().numpy())
    #         accuracy = torch.mean(((output > .5) == (y > .5)).float())
    #         accuracies.append(accuracy.detach().cpu().numpy())
    #
    #     return(np.mean(losses), np.mean(accuracies))
    #
    #
    # """## Training
    # Ok let's train! We'll keep track of validation loss to do [early stopping](https://en.wikipedia.org/wiki/Early_stopping).
    # """
    #
    # train_accs = []
    # val_accs = []
    # patience = 10  # for early stopping
    # patience_counter = patience
    # best_val_loss = np.inf
    # # to save the best model fit to date
    # check_point_filename = 'cnn_1d_checkpoint.pt'
    # for epoch in range(100):
    #     start_time = timeit.default_timer()
    #     train_loss, train_acc = run_one_epoch(
    #         True, train_dataloader, cnn_1d, optimizer, device)
    #     val_loss, val_acc = run_one_epoch(
    #         False, validation_dataloader, cnn_1d, optimizer, device)
    #     train_accs.append(train_acc)
    #     val_accs.append(val_acc)
    #     if val_loss < best_val_loss:
    #         torch.save(cnn_1d.state_dict(), check_point_filename)
    #         best_val_loss = val_loss
    #         patience_counter = patience
    #     else:
    #         patience_counter -= 1
    #         if patience_counter <= 0:
    #             # recover the best model so far
    #             cnn_1d.load_state_dict(torch.load(check_point_filename))
    #             break
    #     elapsed = float(timeit.default_timer() - start_time)
    #     print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
    #           (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter))
    #
    # """Rather than trying to stare at those numbers let's plot training and validation accuracy:"""
    #
    # plt.plot(train_accs, label="train")
    # plt.plot(val_accs, label="validation")
    # plt.legend()
    # plt.grid(which="both")
    #
    # """Unusually the validation accuracy is higher than the train loss in some places! This could be due to (at least) two factors:
    # 1. Relatively small validation set so the estimate is noisy.
    # 2. Dropout introduces noise into the predictions during training but not validation (as it should).
    # We could actually test this by evaluating on the training data without dropout:
    # """
    #
    # train_loss, train_acc = run_one_epoch(
    #     False, train_dataloader, cnn_1d, optimizer, device)
    # val_loss, val_acc = run_one_epoch(
    #     False, validation_dataloader, cnn_1d, optimizer, device)
    # print(train_acc, val_acc)
    #
    # """As expected, the validation accuracy is a little lower.
    #
    # ## Model interpretation
    #
    # Model interpretation is an active area of research for CNNs. Two baseline approaches in genomics are *in silico* mutagenesis and saliency maps. Good approaches exist for explaining the prediction for a single instance: explaining how the model works globally is still a challenge in general.
    #
    # ### in silico mutatgenesis
    #
    # This approach is specific to genomics where we can imagine making individual "point" mutations (changing just one base) and seeing what effect that has on the model's prediction. In computer vision there isn't a direct analogy: deleting or changing a single pixel would rarely (if ever) change the prediction we would expect (although CNNs are not necessarily robust to such changes - adversarial training attempts to make them so).
    #
    # *in silico* mutagenesis can be used both for model interpretation (as we do here) and predicting the effects of real mutations/genetic differences between individuals.
    #
    # We'll just run mutagenesis for the first batch of the validation data:
    # """
    #
    # torch.set_grad_enabled(False)
    # for (x_cpu, y_cpu) in validation_dataloader:
    #     x = x_cpu.to(device)
    #     y = y_cpu.to(device)
    #     output = cnn_1d(x).squeeze()
    #     output = torch.sigmoid(output)
    #     delta_output = torch.zeros_like(x, device=device)
    #     # loop over all positions changing to each position nucleotide
    #     # note everything is implicitly parallelized over the batch here
    #     for seq_idx in range(cnn_1d.seq_len):  # iterate over sequence
    #         for nt_idx in range(4):  # iterate over nucleotides
    #             x_prime = x.clone()  # make a copy of x
    #             x_prime[:, :, seq_idx] = 0.  # change the nucleotide to nt_idx
    #             x_prime[:, nt_idx, seq_idx] = 1.
    #             output_prime = cnn_1d(x_prime).squeeze()
    #             output_prime = torch.sigmoid(output_prime)
    #             delta_output[:, nt_idx, seq_idx] = output_prime - output
    #     break  # just do this for first batch
    #
    # """Note how computationally expensive this is: for every instance we do inference $4 \times L$ times (we could make this $3 \times L$ easily enough).
    #
    # We'll visualize just four (2 positive 2 negative) examples:
    # """
    #
    # delta_output_np = delta_output.detach().cpu().numpy()
    # delta_output_np -= delta_output_np.mean(1, keepdims=True)
    # output_np = output.detach().cpu().numpy()
    # for i in range(1, 5):
    #     pwm_df = pd.DataFrame(
    #         data=delta_output_np[i, :, :].transpose(), columns=("A", "C", "G", "T"))
    #     crp_logo = logomaker.Logo(pwm_df)  # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    #     plt.title("True label: %i. Prob(y=1)=%.3f" % (y_cpu[i], output_np[i]))
    #     plt.show()
    #
    # """For the two positive examples there are clear regions in the sequence that if disrupted significantly impact the prediction. For the negative examples there is a scattering of "mutations" that would effect the prediction, presumably by randomly introducing a sequence that looks a little like the CTCF motif (we don't expect any specific features in these negative sequences).
    #
    # ### Saliency maps
    #
    # This is the simplest approach to leverage the same backprop machinery that we use during training. The trick is that instead of taking the gradient w.r.t. parameters we'll now take the gradient w.r.t. inputs `x`. Specifically here we'll get the gradient of $P(y=1|x)$ w.r.t x. The `saliency` itself is defined as that gradient elementwise multiplied with `x` itself.
    # """
    #
    # torch.set_grad_enabled(True)
    # x.requires_grad_()  # tell torch we will want gradients wrt x (which we don't normally need)
    # output = cnn_1d(x).squeeze()
    # output = torch.sigmoid(output)
    # # in a multiclass model this would be a one-hot encoding of y
    # dummy = torch.ones_like(output)
    # output.backward(dummy)
    # gradient_np = x.grad.detach().cpu().numpy()
    # output_np = output.detach().cpu().numpy()
    # saliency = gradient_np * x_cpu.numpy()
    # for i in range(1, 5):
    #     pwm_df = pd.DataFrame(
    #         data=saliency[i, :, :].transpose(), columns=("A", "C", "G", "T"))
    #     logomaker.Logo(pwm_df)  # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    #     plt.title("True label: %i. Prob(y=1)=%.3f" % (y_cpu[i], output_np[i]))
    #     plt.show()
    #
    # """This is much more computationally efficient than in silico mutagenesis, requiring only ONE forward and backward pass. Compare the two interpretation methods: for the positive examples the same sequence regions are highlighted (although the saliency map is noisier outside that region).
    #
    # ### Other interpretation approaches
    #
    # There's a nice compedium and associated `torch` code for a number methods [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations). Some that I would add:
    # 1. DeepLIFT  [paper](https://arxiv.org/abs/1704.02685) [code](https://github.com/kundajelab/deeplift) - no `pytorch` support sadly.
    # 2. DeepSHAP [paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) [code](https://github.com/slundberg/shap)
    # 3. Influence functions. [paper](https://arxiv.org/abs/1703.04730) [code](https://github.com/kohpangwei/influence-release). A little different: finds which training points most strongly influence the current prediction.
    #
    # # Tasks
    #
    # Make your own copy of this notebook and complete the following questions by filling in the code and write-up sections. Feel free to add cells as needed.
    #
    # You might find it helpful to make a wrapper function that does the following:
    # 1. Instantiates a new `CNN_1d` model (and transfers it to the GPU).
    # 1a. (optional) If you're changing settings that change `cnn_1d.seq_len` or `batchsize` then make new `BedPeakDataset` and `DataLoader` objects with appropriate settings.
    # 2. Instantiates a new optimizer for the model.
    # 3. Runs the training loop with early stopping.
    # 4. Returns the fitted model, train and validation accuracies.
    # """
    #
    #
    #
    #
    # """## Question 1 [4 points]
    #
    # a. Find settings of `CNN_1d` that underfit (low train and test accuracy). [3 points]
    # """
    #
    # """
    # Original settings:
    # CNN_1d(
    #     n_output_channels = 1,
    #     filter_widths = [15, 5],
    #     num_chunks = 5,
    #     max_pool_factor = 4,
    #     nchannels = [4, 32, 32],
    #     n_hidden = 32,
    #     dropout = 0.2,
    # )
    # """
    # cnn_1d = CNN_1d(
    #     n_output_channels=1,
    #     filter_widths=[3],
    #     num_chunks=15,
    #     max_pool_factor=4,
    #     nchannels=[4, 16],
    #     n_hidden=32,
    #     dropout=0.2,
    # )
    # print("Using sequences of length", cnn_1d.seq_len)
    # model, train_accs, val_accs = create_and_train_model(cnn_1d=cnn_1d)
    #
    # plt.plot(train_accs, label="train")
    # plt.plot(val_accs, label="validation")
    # plt.legend()
    # plt.grid(which="both")
    #
    # """b. Describe the setting choices you have made to underfit the data and explain why these settings contributed to a low train and test accuracy. [1 point]
    #
    # *Fill in with your explanation. Feel free to add any plots or tables if you feel they will be helpful.*
    #
    # While there are presumably many ways to get the network to underfit, some of them seemed "unfair" or "overkill", e.g. getting rid of all the layers except the input and output layer. My goal with my modifications was to make as few as possible atomic changes to the network that would still lower validation accuracy below .75. I'd say I only partially succeeded at this, as I had to reduce the network's capacity in multiple ways to get validation accuracy below .75.
    #
    # I achieved my goal by making three changes in sequence.
    #
    # First, I reduced the network's capacity to learn inter-motif relationships by getting rid of the second convolutional layer and just having the first layer of filters feed into the dense layer. My intuition was that this would reduce the network's ability to recognize the inter-motif dynamics Professor Knowles mentioned in class. Specifically, that it would prevent the presence of two motifs from being identified as a higher-level "motif". This change lowered validation accuracy down to 0.8489, which was lower than the original validation accuracy but still not low enough.
    #
    # Second, since an earlier cell's text mentioned that CTCF has a 14nt consensus motif, I "cheated" and tried reducing the filter width of the first (and now only conv layer) to 5 in the hope that this would prevent the network from recognizing binding sites altogether. Frankly, I was surprised to find that this only lowered validation accuracy to 0.7961, which is higher than I'd expect given that a single filter in the network can no longer represent the entire consensus motif. My intuition for why this step didn't reduce performance *more* is that the network might be learning different components of the consensus sequence with different filters, i.e. the distributed representations people talk about CNNs learning.
    #
    # Finally, to get validation accuracy below .75, I reduced filter width further to 3 and lowered the number of filters to 16, which gave a validation accuracy of 0.7369 and training accuracy of .7432. This presumably reduced accuracy further via one of (or both of) two potential mechanisms:
    # 1. Reduce the overall "motif capacity" of the network. That is, if we assume the convolutional layer has some ability to spread its knowledge of motifs across multiple filters, keeping the same number of filters and reducing their width lowers the number of "slots" available for tracking motifs.
    # 2. Reduce the expressivity of individual filters to capture complex motifs, independent of the overall "motif capacity".
    #
    # I explore the influence of these two mechanisms in detail in my answer to Question 3.
    #
    # To summarize, I found the network surprisingly hard to under-fit, perhaps because we're only predicting binding of one TF and our data's not *that* big?
    #
    # ## Question 2 [4 points]
    #
    # a. Find settings of `CNN_1d` that that overfit (high train accuracy but low test accuracy). [3 points]
    # """
    #
    # cnn_1d = CNN_1d(
    #     n_output_channels=1,
    #     filter_widths=[15, 5],
    #     num_chunks=5,
    #     max_pool_factor=4,
    #     nchannels=[4, 256, 32],
    #     n_hidden=32,
    #     dropout=0.,
    # )
    # # print("Using sequences of length", cnn_1d.seq_len)
    # model, train_accs, val_accs = create_and_train_model(cnn_1d=cnn_1d)
    #
    # """b. Describe the setting choices you have made to overfit the data and explain why these settings contributed to a high train and low test accuracy. [1 point]
    #
    # *Fill in with your explanation. Feel free to add any plots or tables if you feel they will be helpful.*
    #
    # I was able to get the network to over-fit a little--validation accuracy of ~.85 vs. training accuracy of ~.94--just by turning off dropout and leaving everything else as-is. One thing that's kind of frightening with respect to this is that the results were very inconsistent (the ones I mention are the least over-fit). Sometimes the dropout-less net would run out of patience at a validation accuracy of around ~.75 and training accuracy of ~.88; other times, the network would achieve results like the ones I mention above, which aren't so far off from the original results. I also find this interesting because it implies that the internet blog posts that claim dropout isn't necessary if you're using batch norm in CNNs are wrong, at least for the dataset/settings we're working with.
    #
    # A more reliable way to get significant over-fitting was to (leave dropout off and) increase the number of convolutional filters in the first layer from 32 to 256. When I did this, I saw training accuracies of as high as .98 and validation accuracies anywhere between .50 and .71. Intuitively, this fits with general ML wisdom that increasing model complexity while reducing regularization leads to over-fitting.
    #
    # Interestingly, re-adding and subsequently increasing dropout let me get better performance than I got with the original network on both validation and training data using many more filters in convolutional layers. As I discuss in Question 4, I leveraged this to achieve relatively high scores on the leaderboard (~.98, which was high at the time of writing) without doing any other sort of tuning.
    #
    # ## Question 3  [6 points]
    #
    # a. Carefully explore varying one architectural choice (depth, number of channels, filter width, regularization, pooling factor, optimizer, learning rate, batch size, activation function, normalization, skip connections). Report the final train and validation accuracy as a function of this choice.
    #
    # For example you might vary the number of convolutional layers from 1 to 4, keeping everything else the same, and plot validation accuracy vs number of layers. [4 points]
    # """
    #
    # filter_capacity = 15 * 32
    # val_accs_by_filter_width = []
    # for filter_width in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
    #     cnn_1d = CNN_1d(
    #         n_output_channels=1,
    #         filter_widths=[filter_width, 5],
    #         num_chunks=5,
    #         max_pool_factor=4,
    #         nchannels=[4, int(filter_capacity / filter_width), 32],
    #         n_hidden=32,
    #         dropout=0.2,
    #     )
    #     model, train_accs, val_accs = create_and_train_model(cnn_1d=cnn_1d)
    #     val_accs_by_filter_width.append((filter_width, max(val_accs)))
    # val_accs_by_filter_width
    #
    # val_accs_by_filter_width
    # plt.plot(
    #     [va[0] for va in val_accs_by_filter_width],
    #     [va[1] for va in val_accs_by_filter_width],
    #     label="validation accuracy")
    # plt.legend()
    #
    # """b. For your selected architectural choice, discuss how and why varying this option affected your training and validation accuracy. [2 points]
    #
    # Since the assignment introduction mentions that using a filter width of 15 is "cheating" because CTCF is known to have a 14 nt consensus motif, I wanted to test whether varying filter width would dramatically impact the network's performance. This seemed interesting from both a biological and machine learning perspective. On the biology side, it's interesting because it's possible the network is discovering smaller/larger motifs that aren't obviously related to the consensus one. On the machine learning side, it's interesting because I've been told that convolutional networks tend to have "distributed" representations, and this seemed like a chance to test that claim.
    #
    # I had a much easier time testing the interesting machine learning hypothesis than I did testing the interesting biology one. In order to test whether the CNN was able to recover good motifs via "distributed representations", I varied filter width while keeping "motif capacity" constant. By motif capacity, I mean the total number of slots in the first layer of convolutional filters. I chose to keep this fixed because it ensures that each version of the network would be able to represent the same motifs in its first layer assuming it could perfectly distribute representations over multiple filters. Put another way, if I had just varied filter width, the smaller filter width networks would've been at an additional disadvantage due to having less overall space in their first layer.
    #
    # When I kept "motif capacity" constant and varied filter width, I found that validation accuracy performance was definitely impacted by filter width but minimally, with the exception of when I made filter width very small (<5). Looking at the above graph of validation accuracy vs. filter width, observe that even with a filter width of 7 the network's validation accuracy was in the high 80s, only a few percentage points below the max validation accuracy of ~.91, found by using a filter width of >=13. I'm actually pleasantly surprised by this as it lends credence to the hypothesis that CNNs are in fact good at learning distributed representations.
    #
    # ## Question 4 [8 points]
    #
    # a. Get as good validation accuracy as you can! [4 points]
    #
    # Optionally you can try some more advanced extensions, e.g.
    # 1. Adding some [Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers). Be warned that `torch` assumes the opposite dimensions for convolutional vs recurrent layers so you'll want to use `torch.transpose` appropriately.
    # 2. Transcription factors can bind to either strand of the DNA so you might want to include the [reverse complement](https://www.bx.psu.edu/old/courses/bx-fall08/definitions.html) in addition to the normally input.
    # 3. Changing `BedPeaksDataset` to generate more negative examples in the space between peaks (although you'd only want to do this on the training data so you keep a fixed validation set).
    # """
    #
    #
    # class CNN_1d(nn.Module):
    #     """CNN that uses the reverse complement in addition to the forward sequence."""
    #
    #     def __init__(self,
    #                  n_output_channels=1,
    #                  filter_widths=[15, 5],
    #                  num_chunks=5,
    #                  max_pool_factor=4,
    #                  nchannels=[4, 32, 32],
    #                  n_hidden=32,
    #                  dropout=0.2):
    #
    #         super(CNN_1d, self).__init__()
    #         self.rf = 0  # receptive field
    #         self.chunk_size = 1  # num basepairs corresponding to one position after convolutions
    #
    #         conv_layers = []
    #         for i in range(len(nchannels)-1):
    #             conv_layers += [nn.Conv1d(nchannels[i], nchannels[i+1], filter_widths[i], padding=0),
    #                             nn.BatchNorm1d(nchannels[i+1]),
    #                             nn.Dropout2d(dropout),
    #                             nn.MaxPool1d(max_pool_factor),
    #                             nn.ELU(inplace=True)]
    #             assert(filter_widths[i] % 2 == 1)  # assume this
    #             self.rf += (filter_widths[i] - 1) * self.chunk_size
    #             self.chunk_size *= max_pool_factor
    #
    #         self.conv_net = nn.Sequential(*conv_layers)
    #
    #         self.seq_len = num_chunks * self.chunk_size + \
    #             self.rf  # amount of sequence context required
    #
    #         print("Receptive field:", self.rf, "Chunk size:", self.chunk_size,
    #               "Number chunks:", num_chunks, "Sequence len: ", self.seq_len)
    #
    #         self.dense_net = nn.Sequential(nn.Linear(nchannels[-1] * num_chunks, n_hidden),
    #                                        nn.Dropout(dropout),
    #                                        nn.ELU(inplace=True),
    #                                        nn.Linear(n_hidden, n_output_channels))
    #
    #     def forward(self, x_f, x_b):
    #         net_f = self.conv_net(x_f)
    #         net_f = net_f.view(net_f.size(0), -1)
    #         net_f = self.dense_net(net_f)
    #
    #         net_b = self.conv_net(x_b)
    #         net_b = net_b.view(net_b.size(0), -1)
    #         net_b = self.dense_net(net_b)
    #
    #         return torch.max(net_f, net_b)
    #
    #
    # cnn_1d = CNN_1d()
    #
    # print("Input length:", cnn_1d.seq_len)
    #
    # cnn_1d
    #
    #
    # def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):
    #
    #     torch.set_grad_enabled(train_flag)
    #     cnn_1d.train() if train_flag else cnn_1d.eval()
    #
    #     losses = []
    #     accuracies = []
    #
    #     for (x_f, x_b, y) in dataloader:
    #         (x_f, x_b, y) = (x_f.to(device), x_b.to(
    #             device), y.to(device))  # transfer data to GPU
    #
    #         output = cnn_1d(x_f, x_b)  # forward pass
    #         output = output.squeeze()  # remove spurious channel dimension
    #         loss = F.binary_cross_entropy_with_logits(
    #             output, y)  # numerically stable
    #
    #         if train_flag:
    #             loss.backward()  # back propagation
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #         losses.append(loss.detach().cpu().numpy())
    #         accuracy = torch.mean(((output > .5) == (y > .5)).float())
    #         accuracies.append(accuracy.detach().cpu().numpy())
    #
    #     return(np.mean(losses), np.mean(accuracies))
    #
    #
    # def create_and_train_model(cnn_1d=CNN_1d(), train_dataset=None, val_dataset=None, epochs=100):
    #     """
    #     Instantiate, train, and return a 1D CNN model and its accuracy metrics.
    #     """
    #     # Reload data
    #     if train_dataset is None:
    #         train_dataset = BedPeaksDataset(train_data, genome, cnn_1d.seq_len)
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=1000, num_workers=0)
    #     if val_dataset is None:
    #         val_dataset = BedPeaksDataset(validation_data, genome, cnn_1d.seq_len)
    #     validation_dataloader = torch.utils.data.DataLoader(
    #         val_dataset, batch_size=1000)
    #
    #     # Set up model and optimizer
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     cnn_1d.to(device)
    #     optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True)
    #
    #     # Training loop w/ early stopping
    #     train_accs = []
    #     val_accs = []
    #     patience = 15  # for early stopping
    #     patience_counter = patience
    #     best_val_loss = np.inf
    #     # to save the best model fit to date
    #     check_point_filename = 'cnn_1d_checkpoint.pt'
    #     for epoch in range(epochs):
    #         start_time = timeit.default_timer()
    #         train_loss, train_acc = run_one_epoch(
    #             True, train_dataloader, cnn_1d, optimizer, device)
    #         val_loss, val_acc = run_one_epoch(
    #             False, validation_dataloader, cnn_1d, optimizer, device)
    #         train_accs.append(train_acc)
    #         val_accs.append(val_acc)
    #         if val_loss < best_val_loss:
    #             torch.save(cnn_1d.state_dict(), check_point_filename)
    #             best_val_loss = val_loss
    #             patience_counter = patience
    #         else:
    #             patience_counter -= 1
    #             if patience_counter <= 0:
    #                 # recover the best model so far
    #                 cnn_1d.load_state_dict(torch.load(check_point_filename))
    #                 break
    #         elapsed = float(timeit.default_timer() - start_time)
    #         print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
    #               (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter))
    #     return cnn_1d, train_accs, val_accs
    #
    #
    # REV = str.maketrans("AGCT", "TCGA")
    #
    #
    # def reverse_complement(seq):
    #     return seq.translate(REV)[::-1]
    #
    #
    # class BedPeaksDatasetWithReverseComplement(torch.utils.data.IterableDataset):
    #
    #     def __init__(self, binding_data, genome, context_length):
    #         super(BedPeaksDatasetWithReverseComplement, self).__init__()
    #         self.context_length = context_length
    #         self.binding_data = binding_data
    #         self.genome = genome
    #
    #     def __iter__(self):
    #         prev_end = 0
    #         prev_chrom = ""
    #         for row in self.binding_data.itertuples():
    #             midpoint = int(.5 * (row.start + row.end))
    #             boundary = self.context_length // 2
    #
    #             seq = self.genome[row.chrom][midpoint -
    #                                          boundary: midpoint + boundary]
    #             forward_seq = torch.from_numpy(one_hot(seq))
    #             backward_seq = torch.from_numpy(one_hot(reverse_complement(seq)))
    #             yield(forward_seq, backward_seq, np.float32(1))  # positive example
    #
    #             if prev_chrom == row.chrom:
    #                 midpoint = int(.5 * (prev_end + row.start))
    #                 seq = self.genome[row.chrom][midpoint -
    #                                              boundary:midpoint + boundary]
    #                 forward_seq = torch.from_numpy(one_hot(seq))
    #                 backward_seq = torch.from_numpy(
    #                     one_hot(reverse_complement(seq)))
    #                 # negative example midway inbetween peaks
    #                 yield(forward_seq, backward_seq, np.float32(0))
    #
    #             prev_chrom = row.chrom
    #             prev_end = row.end
    #
    # # best_config = None
    # # max_val_acc = 0
    # # best_model = None
    #
    # # for l1_filters in [512, 1024]:
    # #     for l2_filters in [512]:
    # #         for l3_filters in [512]:
    # #             for n_hidden in [512]:
    # #                 cnn_1d = CNN_1d(
    # #                     n_output_channels = 1,
    # #                     filter_widths = [filter_width, 5, 3],
    # #                     num_chunks = 5,
    # #                     max_pool_factor = 2,
    # #                     nchannels = [4, l1_filters, l2_filters, l3_filters],
    # #                     n_hidden = n_hidden,
    # #                     dropout = 0.5,
    # #                 )
    # #                 print("L1 filters: %d, L2 filters: %d, L3 filters: %d, N_hidden: %d" % (l1_filters, l2_filters, l3_filters, n_hidden))
    # #                 model, train_accs, val_accs = create_and_train_model(cnn_1d=cnn_1d)
    #
    # #                 print("max validation acc for this run: %f" % max(val_accs))
    #
    # #                 if max(val_accs) > max_val_acc:
    # #                     max_val_acc = max(val_accs)
    # #                     best_config = (l1_filters, l2_filters, l3_filters,  n_hidden)
    # #                     best_model = cnn_1d
    #
    # #             torch.cuda.empty_cache()
    #
    #
    # cnn_1d = CNN_1d(
    #     n_output_channels=1,
    #     filter_widths=[15, 7, 5],
    #     num_chunks=5,
    #     max_pool_factor=4,
    #     nchannels=[4, 256, 512, 512],
    #     n_hidden=512,
    #     dropout=0.5,
    # )
    #
    # train_dataset = BedPeaksDatasetWithReverseComplement(
    #     train_data, genome, cnn_1d.seq_len)
    # validation_dataset = BedPeaksDatasetWithReverseComplement(
    #     validation_data, genome, cnn_1d.seq_len)
    #
    # model, train_accs, val_accs = create_and_train_model(
    #     cnn_1d=cnn_1d,
    #     train_dataset=train_dataset,
    #     val_dataset=validation_dataset
    # )
    #
    # """b. What gave you the biggest boost in performance? Why do you think that is? [2 points]
    #
    # ## Part 1: Hyperparameter Tuning
    #
    # Before trying any of the domain-specific approaches, I was able to get a score of 0.98466 on Kaggle and a validation accuracy of ~96.5 just by doing an ad-hoc hyperparameter search (using for-loops, I couldn't get any of the grid search libraries I found to work in Colab with PyTorch) over channel sizes and CNN layer counts. (Note: I left this code commented out since it takes really long to run.)
    #
    # After increasing dropout rate to 0.5 based on reading in a few places that having a %50 dropout rate often worked well for bigger nets, my hyper-parameter search found that validation accuracy was maximized by using 256, 512, and 512 convolutional filters in the first, second, and third layers respectively.
    #
    # Frankly, I could concoct some post-hoc story for why this makes sense, but I was honestly surprised to find that having fewer layers in the first layer than in the second and third layers performed better than the inverse. The typical story is that having multiple convolutional layers allows us to slowly downsample and learn better higher-level representations. But that would lead us to predict that you always want more filters in the first layer than the second (and so on for subsequent layers). One pretty hand-wavy thing I can imagine being true is that the first layer of binding motifs are relatively easy to learn but the non-adjacent relationships between them that predict binding for the trickier cases (i.e. the ones that get you to really high accuracy)
    #
    # Slightly less specific, clearly adding more filters in general gave me a large performance boost. **This** makes logical sense if we assume that even a single TF like CTCF actually has a quite complicated binding pattern that benefited from more than the initial 32 filters.
    #
    # ## Part 2: Reverse Complement
    #
    # From there, I changed the code to train the CNN using the forward and reverse complement versions of the sequence and return the maximum of the two forward passes' outputs. This got me a small performance boost, but honestly small enough that I question whether it was statistically significant. In theory, this could have helped performance by letting the model notice patterns in the reverse complement sequences that influence TF binding (as Professor Knowles discussed in class). But again, I'd have to do more investigation, likely in a use-case where accuracy wasn't already so high, to confirm that hypothesis.
    #
    # c.  Use some model interpretability technique to visualize why the model has made the assignments it did for a few examples. This can be one of the methods shown above (in silico mutagenesis or saliency maps) or one of the methods linked under *Other interpretation approaches*. [2 points]
    # """
    #
    # validation_dataloader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=1000)
    #
    # exs_to_visualize = []
    # i = 0
    # for (ex_f, ex_b, ex_y) in validation_dataloader:
    #     torch.set_grad_enabled(True)
    #     cnn_1d_cpu = cnn_1d.to("cpu")
    #     # tell torch we will want gradients wrt x (which we don't normally need)
    #     ex_f.requires_grad_()
    #     # tell torch we will want gradients wrt x (which we don't normally need)
    #     ex_b.requires_grad_()
    #
    #     output_f = torch.sigmoid(cnn_1d(ex_f, ex_f).squeeze())
    #     output_b = torch.sigmoid(cnn_1d(ex_b, ex_b).squeeze())
    #     output_c = torch.max(output_f, output_b)
    #
    #     dummy_f = torch.ones_like(output_f)
    #     dummy_b = torch.ones_like(output_b)
    #
    #     output_f.backward(dummy_f)
    #     output_b.backward(dummy_b)
    #
    #     gradient_f_np = ex_f.grad.detach().cpu().numpy()
    #     gradient_b_np = ex_b.grad.detach().cpu().numpy()
    #     gradient_c_np = gradient_b_np + gradient_f_np
    #
    #     output_np = output.detach().cpu().numpy()
    #     saliency_f = gradient_f_np * ex_f.detach().numpy()
    #     saliency_b = gradient_b_np * ex_b.detach().numpy()
    #     saliency_c = saliency_f + saliency_b
    #     for i in range(1, 5):
    #         pwm_f_df = pd.DataFrame(
    #             data=saliency_f[i, :, :].transpose(), columns=("A", "C", "G", "T"))
    #         pwm_b_df = pd.DataFrame(
    #             data=saliency_b[i, :, :].transpose(), columns=("A", "C", "G", "T"))
    #         pwm_c_df = pd.DataFrame(
    #             data=saliency_c[i, :, :].transpose(), columns=("A", "C", "G", "T"))
    #
    #         logomaker.Logo(pwm_f_df)  # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    #         plt.title("[Forward] True label: %i. Prob(y=1)=%.3f" %
    #                   (ex_y[i], output_np[i]))
    #         plt.show()
    #
    #         logomaker.Logo(pwm_b_df)  # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    #         plt.title("[Backward] True label: %i. Prob(y=1)=%.3f" %
    #                   (ex_y[i], output_np[i]))
    #         plt.show()
    #
    #         logomaker.Logo(pwm_c_df)  # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    #         plt.title("[Combined] True label: %i. Prob(y=1)=%.3f" %
    #                   (ex_y[i], output_np[i]))
    #         plt.show()
    #     break
    #
    # """Plotting a few saliency maps for some samples was surprising useful for understanding at least one aspect of the network's behavior. If you look at the above saliency maps, you'll notice that, besides that the network assigns very high probabilities in either direction, all of the positively classified examples have at least one densely clustered area with a combination of spiky positive and negative probability sequence expectations at different positions. If you squint at a few of the positive examples, you can even see a pattern in the nucleotides that get assigned high and low probability.
    #
    # Another thing this helped show was that the reverse complement inputs were sometimes used to determine the outputs even though adding them didn't make a huge difference performance-wise. You can see this because for a few of the examples, the "[Combined]" logo matches the "[Backward]" one, which means that the backward seq output was bigger than the forward seq output (this claim falls out of the way back-propagation treats maxes).
    #
    # ### Question 5  [6 points]
    #
    # The final task is to submit your best performing model's predictions on chromosome 1 to [our Kaggle InClass competition](https://www.kaggle.com/c/predict-ctcf-binding/). We'll show how to do this with the example model so you can do it for yours.
    #
    # You can change what train/validation split you use here also if you want: e.g. you could do K-fold cross-validation or even retrain including the validation data if you think it will help (although early stopping may not work any more).
    #
    # First we'll need to load the test regions:
    # """
    #
    # test_data = pd.read_csv(DATADIR + "CTCF_A549_test_masked.bed.gz",
    #                         sep='\t', names=("chrom", "start", "end", "name", "score"))
    # test_data.head(10)
    #
    # """Unlike the training data we've included random (in number and in genomic position) negative (no binding) regions in this bed file since otherwise you'd know implicitly that all the loaded regions are positives!
    #
    # We'll use a new `Dataset` class and `DataLoader` to match predictions on the test data in batches:
    # """
    #
    #
    # class BedPeaksDatasetTest(torch.utils.data.IterableDataset):
    #
    #     def __init__(self, binding_data, genome, context_length):
    #         super(BedPeaksDatasetTest, self).__init__()
    #         self.context_length = context_length
    #         self.binding_data = binding_data
    #         self.genome = genome
    #
    #     def __iter__(self):
    #         for row in self.binding_data.itertuples():
    #             midpoint = int(.5 * (row.start + row.end))
    #             seq = self.genome[row.chrom][midpoint -
    #                                          self.context_length//2:midpoint + self.context_length//2]
    #             yield(one_hot(seq), one_hot(reverse_complement(seq)))
    #
    #
    # test_dataset = BedPeaksDatasetTest(test_data, genome, cnn_1d.seq_len)
    # # you can always use a smaller batchsize if you ended up using a really big model
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1000, num_workers=0)
    #
    # model = model.to(device)
    # outputs = []
    # for x_f, x_b in test_dataloader:  # iterate over batches
    #     x_f, x_b = x_f.to(device), x_b.to(device)
    #     output = model(x_f, x_b).squeeze()
    #     output = torch.sigmoid(output)
    #     output_np = output.detach().cpu().numpy()
    #     outputs.append(output_np)
    # output_np = np.concatenate(outputs)
    #
    # my_solution = pd.DataFrame(
    #     data={"Id": range(len(output_np)), "Predicted": output_np})
    # # save this somewhere else if you want
    # my_solution.to_csv("my_solution_final.csv", index=False)
    # my_solution.head()
    #
    # """Now you can go to https://www.kaggle.com/c/predict-ctcf-binding/ and make a submission!
    # 1. Sign into `kaggle` via your Columbia Google account.
    # 2. Make sure your team name is your actual name so we know which is your submission.
    # 3. Download `my_submission.csv` using the `Files` browser on the left.
    # 4. Upload your submission and see where you get on the leaderboard!
    #
    # For the assignment you only need to upload one submission, but you can upload two submissions per day until the deadline if you want!
    #
    # Submission is worth 2 points, beating our baseline (auPR=0.94) gets you 2 points and the other 2 points are for your tertile in the class (e.g. you get 2 points if you're in the top third).
    #
    # I'm going to think about prizes for first and second place!
    # """

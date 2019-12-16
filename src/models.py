import torch
import torch.nn as nn


class CNN_1d(nn.Module):
    def __init__(
        self,
        n_output_channels=1,
        filter_widths=[15, 5],
        num_chunks=5,
        max_pool_factor=4,
        nchannels=[4, 32, 32],
        n_hidden=32,
        dropout=0.2,
    ):

        super(CNN_1d, self).__init__()
        self.rf = 0  # receptive field
        self.chunk_size = (
            1  # num basepairs corresponding to one position after convolutions
        )

        conv_layers = []
        for i in range(len(nchannels) - 1):
            conv_layers += [
                nn.Conv1d(nchannels[i], nchannels[i + 1], filter_widths[i], padding=0),
                nn.BatchNorm1d(nchannels[i + 1]),
                nn.Dropout2d(dropout),
                nn.MaxPool1d(max_pool_factor),
                nn.ELU(inplace=True),
            ]
            assert filter_widths[i] % 2 == 1  # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.chunk_size *= max_pool_factor

        self.conv_net = nn.Sequential(*conv_layers)

        self.seq_len = (
            num_chunks * self.chunk_size + self.rf
        )  # amount of sequence context required

        print(
            "Receptive field:",
            self.rf,
            "Chunk size:",
            self.chunk_size,
            "Number chunks:",
            num_chunks,
            "Sequence len: ",
            self.seq_len,
        )

        self.dense_net = nn.Sequential(
            nn.Linear(nchannels[-1] * num_chunks, n_hidden),
            nn.Dropout(dropout),
            nn.ELU(inplace=True),
            nn.Linear(n_hidden, n_output_channels),
        )

    def forward(self, x, with_sigmoid=False):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        if with_sigmoid:
            net = torch.sigmoid(net)
        return net


def get_default_cnn():
    return CNN_1d(
        num_chunks=2,
        filter_widths=[15, 5, 3],
        nchannels=[4, 128, 64, 64],
        n_hidden=64,
        dropout=0.5,
    )

def get_big_cnn():
    return CNN_1d(
        num_chunks=2,
        filter_widths=[15, 7, 3],
        nchannels=[4, 256, 128, 64],
        n_hidden=64,
        dropout=0.5,
    )

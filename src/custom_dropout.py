import torch
from torch import nn


class LockedDropout(nn.Module):
    """
    LockedDropout is a base class for applying the same dropout mask to every input in a mini-batch.

    This class is based off of the implementation included in the pytorch NLP docs:
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`.
    Here is their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.
    """

    def __init__(self, p=0.1, training=True):
        """

        Args:
            p (float): Probability of an element in the dropout mask to be zeroed.
            training (bool): Is the network currently being trained or used for prediction?
        """
        self.p = p
        self.training = training
        super().__init__()

    def train(self, training=True):
        self.training = training

    @classmethod
    def _get_mask_shape(cls, x):
        raise NotImplementedError("Don't instantiate LockedDropout directly")

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [batch size, batch size, rnn hidden size]):
                Input to apply dropout to.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(*self._get_mask_shape(x), requires_grad=False)
        mask = mask.bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)  # rescaling
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + "p=" + str(self.p) + ")"


class LockedWeightDropout(LockedDropout):
    @classmethod
    def _get_mask_shape(cls, x):
        return x.new_empty(1, x.size(1), x.size(2), x.size(3), requires_grad=False)


class LockedChannelDropout(LockedDropout):
    """
    LockedChannelDropout's dropout mask zeros out an entire channel of the input in a mini-batch.
    """

    def __init__(self, p=0.1, training=True):
        super().__init__(p, training=training)

    @classmethod
    def _get_mask_shape(cls, x):
        return (1, x.size(1), 1, 1)


def replace_dropout_layers(model, dropout_cls=LockedChannelDropout):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = replace_dropout_layers(module)

        if type(module) == nn.Dropout:
            model._modules[name] = dropout_cls(p=module.p, training=module.training)

    return model


def apply_dropout(m):
    if type(m) == nn.Dropout or isinstance(m, LockedDropout):
        m.train()


def unapply_dropout(m):
    if type(m) == nn.Dropout or isinstance(m, LockedDropout):
        m.eval()

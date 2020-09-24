import numpy as np


def abs_max(x, axis): 
    x_max, x_min = np.max(x, axis=axis), np.min(x, axis=axis)
    return np.where(-x_min > x_max, x_min, x_max)


def rolling_window(a, window):
    """
    Take an array and a window size and split the elements in the array into overlapping windows.

    Note: This involves some numpy black magic shit. I didn't invent it; I stole it from
    here (https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html). That said,
    I spent way too much time understanding how it works to not document it, realistically
    for my future self, so here we go...

    The high-level idea is we take the final axis of a numpy array and chunk it into overlapping
    windows. However, rather than just iterate via a for-loop and duplicate the chunks of the array
    we want, we leave the original array's in-memory structure as-is and change what happens
    when we index into it. This is where `strides` come in.
    """
    # Break the last axis into a series of windows
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    # Create a new view with duplicated overlapping `window`-sized chunks
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def convolve_1d(a, kernel):
    """
    Arguments:
        a: np.ndarray
            Assumed to have shape (B, K, N), which (see below) should be compatible with `kernel`'s
            dimensions.
        kernel: np.ndarray
            Should have shape (K, L) where L <= N and K matches `a`'s penultimate dimension.
    """
    # output shape: (B, L - K + 1, K)
    strided_a = rolling_window(a, kernel.shape[-1]) 
    # Add dummy dimensions to kernel for broadcasting
    return np.sum(strided_a * kernel[None, :, None, :], axis=(1, 3))

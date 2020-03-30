import numpy as np

def abs_max(x, axis): 
    x_max, x_min = np.max(x, axis=axis), np.min(x, axis=axis)
    return np.where(-x_min > x_max, x_min, x_max)


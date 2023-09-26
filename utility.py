import numpy as np


def is_number(x):
    return isinstance(x, int) or isinstance(x, float)


def numpyze(x):
    if is_number(x):
        return np.array([x])
    else:
        return np.array(x)

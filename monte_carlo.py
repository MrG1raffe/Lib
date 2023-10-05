import numpy as np
from scipy.stats import norm
from typing import Tuple
from numpy.typing import NDArray
from numpy import float_


"""
ToDo: rewrite as a class.
"""


def monte_carlo(
    sample: NDArray[float_],
    proba: float = 0.95,
    axis: int = 0
) -> Tuple:
    """
    Calculates the mean of the sample and confidence intervals of level 'proba'.

    Args:
        sample: array to compute the mean.
        proba: confidence level for the intervals.
        axis: axis along which the mean to be computed

    Returns:
        mean, ci_dev: empirical mean 'mean' and the deviation of the confidence interval, i.e. ci = (mean - ci_dev, mean + ci_dev).
    """
    mean = np.mean(sample, axis=axis)
    var = np.var(sample, ddof=1, axis=axis)
    quantile = norm.ppf((1 + proba) / 2)
    ci_dev = quantile * np.sqrt(var / sample.shape[axis])
    return mean, ci_dev

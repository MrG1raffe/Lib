import numpy as np
from typing import Union, Any
from numpy.typing import NDArray
from numpy import float_

DEFAULT_SEED = 42


def is_number(x: Any) -> bool:
    """
    Checks whether x is int or float.
    """
    return isinstance(x, int) or isinstance(x, float)


def numpyze(x: Any) -> NDArray[float_]:
    """
    Converts x to numpy array.
    """
    if is_number(x):
        return np.array([x])
    else:
        return np.array(x)


def get_vanilla_payoff(
        K: Union[float, NDArray[float_]],
        flag: str
):
    """
    Returns the payoff of the vanilla Call/Put option with strike 'K'.

    Args:
        K: strikes.
        flag: 'c' for calls, 'p' for puts.

    Returns:
        Call or put payoff as a function of the underlying price.
    """
    def call_payoff(x):
        return np.maximum(x - K, 0)

    def put_payoff(x):
        return np.maximum(K - x, 0)

    return call_payoff if flag == 'c' else put_payoff

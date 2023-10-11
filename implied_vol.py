from typing import Union
from numpy.typing import NDArray
from numpy import float_
import numpy as np
from BlackScholes import BlackScholes
from py_vollib.black.implied_volatility import implied_volatility
iv_lets_be_rational = np.vectorize(implied_volatility)


def newton_init_vol(
    T: Union[float, NDArray[float_]],
    K: Union[float, NDArray[float_]],
    F: Union[float, NDArray[float_]]
) -> Union[float, NDArray[float_]]:
    """
    Returns initial volatility for Newton's method.

    Args:
        T: times to maturity.
        K: strikes.
        F: forward prices at t = 0.

    Returns:
        Initial volatility.
    """
    return np.sqrt(2 / T * np.abs(np.log(F / K)))


def black_iv(
        option_price: Union[float, NDArray[float_]],
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
        r: Union[float, NDArray[float_]],
        flag: str,
        method: str = 'LetsBeRational',
        tol: float = 1e-8
) -> Union[float, NDArray[float_]]:
    """
    Calculates implied vol in the Black-76 model given the option price and parameters.

    Args:
        option_price: option prices.
        T: times to maturity.
        K: strikes.
        F: forward prices at t = 0.
        r: the risk-free interest rate.
        flag: 'c' for calls, 'p' for puts.
        method: which method to use for the calculation: 'LetsBeRational' or 'Newton'.
        tol: tolerance for the Newton's method.

    Returns:
        Implied volatility or an array of implied volatilities corresponding to the prices.
    """
    if method == 'LetsBeRational':
        return iv_lets_be_rational(option_price, F, K, r, T, flag)
    if method == 'Newton':
        sigma = newton_init_vol(T, K, F)
        step = np.inf
        while np.max(np.abs(step)) > tol:
            model = BlackScholes(sigma=sigma, r=r)
            step = (model.vanilla_price(T, K, F / np.exp(r * T), flag) - option_price) / model.vega(T, K, F)
            sigma -= step
        return sigma
    raise ValueError("Wrong method name. Choose either 'LetsBeRational' or 'Newton'.")


def black_scholes_iv(
        option_price: Union[float, NDArray[float_]],
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
        r: Union[float, NDArray[float_]],
        flag: str,
        method: str = 'LetsBeRational',
        tol: float = 1e-8
) -> Union[float, NDArray[float_]]:
    """
    Calculates implied vol in the Black-Scholes model given the option price and parameters.

    Args:
        option_price: option prices.
        T: times to maturity.
        K: strikes.
        F: spot prices at t = 0.
        r: the risk-free interest rate.
        flag: 'c' for calls, 'p' for puts.
        method: which method to use for the calculation: 'LetsBeRational' or 'Newton'.
        tol: tolerance for the Newton's method.

    Returns:
        Implied volatility or an array of implied volatilities corresponding to the prices.
    """
    F = S * np.exp(r * T)
    return black_iv(option_price, T, K, F, r, flag, method, tol)

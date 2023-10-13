import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from typing import Union, Tuple
from numpy.typing import NDArray
from numpy import float_

np.seterr(divide='ignore')


@dataclass
class BlackScholes():
    """
    Will not be updated. Use Black76 instead!
    """
    sigma: Union[float, NDArray] = None  # volatility in 1-d model or covariance matrix
    r: float = 0

    def _d1d2(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]]
    ) -> Tuple[Union[float, NDArray[float_]]]:
        """
        Calculates d1 and d2 from Black-Scholes formula.

        Args:
            T: times to maturity.
            K: strikes.
            S: spot prices at t = 0.

        Returns:
            d1: values of d1.
            d2: values of d2.
        """
        d1 = (np.log(S/K) + (self.r + self.sigma**2/2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2

    def vanilla_price(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the vanilla option price via Black-Scholes formula

        Args:
            T: times to maturity.
            K: strikes.
            S: spot prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Prices of the call/put vanilla options.
        """
        d1, d2 = self._d1d2(T, K, S)
        if flag == 'c':
            return S * norm.cdf(d1) - np.exp(-self.r*T) * K * norm.cdf(d2)
        if flag == 'p':
            return -S * norm.cdf(-d1) + np.exp(-self.r*T) * K * norm.cdf(-d2)

    def delta(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option delta in the Black-Scholes model

        Args:
            T: times to maturity.
            K: strikes.
            S: spot prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, S)
        return norm.cdf(d1) if flag == 'c' else -norm.cdf(-d1)

    def gamma(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option gamma in the Black-Scholes model

        Args:
            T: times to maturity.
            K: strikes.
            S: spot prices at t = 0.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, S)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

    def vega(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option vega in the Black-Scholes model

        Args:
            T: times to maturity.
            K: strikes.
            S: spot prices at t = 0.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, S)
        return S * norm.pdf(d1) * np.sqrt(T)

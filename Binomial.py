import numpy as np
from dataclasses import dataclass
from typing import Union, Callable
from numpy.typing import NDArray
from numpy import float_
from scipy.special import comb


@dataclass
class Binomial():
    u: float = None
    d: float = None
    r: float = 0

    def _risk_neutral_proba(self):
        p = (self.r - self.d) / (self.u - self.d)
        q = 1 - p
        return p, q

    def european_option_price(
        self,
        T: Union[float, NDArray[float_]],
        S: float,
        payoff: Callable
    ):
        n = np.arange(T + 1)
        p, q = self._risk_neutral_proba()
        S_T = S * (1 + self.u)**n * (1 + self.d)**(T - n)
        return np.sum(comb(T, n) * (p**n) * (q**(T - n)) * payoff(S_T)) / (1 + self.r)**T

    def vanilla_price(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
        flag: str
    ):
        def call_payoff(x):
            return np.maximum(x - K, 0)

        def put_payoff(x):
            return np.maximum(K - x, 0)

        payoff = call_payoff if flag == 'c' else put_payoff
        return self.european_option_price(T, S, payoff)

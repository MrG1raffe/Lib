import numpy as np
from dataclasses import dataclass
from typing import Union, Callable, Tuple
from numpy.typing import NDArray
from numpy import float_
from scipy.special import comb


@dataclass
class Binomial():
    """
    Binomial model with risk-free rate 'r' and risky asset rates 'u' and 'd' with u > d.
    """
    u: float = None
    d: float = None
    r: float = 0

    def _risk_neutral_proba(
            self
    ) -> Tuple:
        """
        Calculates risk-neutral probabilities in the model.

        Returns:
            (p, q), where p is the risk-neutral probability of moving up and q for moving down.
        """
        p = (self.r - self.d) / (self.u - self.d)
        q = 1 - p
        return p, q

    def european_option_price(
        self,
        T: Union[float, NDArray[float_]],
        S: float,
        payoff: Callable
    ) -> float:
        """
        Calculates price of the European option with arbitrary payoff.

        Args:
            T: time to maturity.
            S: spot price at t = 0.
            payoff: a function of S determining the payoff at the maturity.

        Returns:
            Option price in the binomial model.
        """
        n = np.arange(T + 1)
        p, q = self._risk_neutral_proba()
        S_T = S * (1 + self.u)**n * (1 + self.d)**(T - n)
        return np.sum(comb(T, n) * (p**n) * (q**(T - n)) * payoff(S_T)) / (1 + self.r)**T

    def vanilla_price(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        S: Union[float, NDArray[float_]],
        flag: str,
        style: str = 'european'
    ) -> float:
        """
        Calculates price of the vanilla Call or Put option.

        Args:
            T: time to maturity.
            K: strike of the option.
            S: spot price at t = 0.
            flag: 'c' for calls, 'p' for puts.
            style: option style, 'european' or 'american'.

        Returns:
            Vanilla option price in the binomial model.
        """
        def call_payoff(x):
            return np.maximum(x - K, 0)

        def put_payoff(x):
            return np.maximum(K - x, 0)

        payoff = call_payoff if flag == 'c' else put_payoff
        if style == 'european':
            return self.european_option_price(T, S, payoff)
        elif style == 'american':
            return self.american_option_price(T, S, payoff)
        else:
            raise ValueError("Wrong 'style' argument was given.")

    def american_option_price(
        self,
        T: float,
        S: float,
        payoff: Callable
    ) -> float:
        """
        Calculates price of the American option with arbitrary payoff.

        Args:
            T: time to maturity.
            S: spot price at t = 0.
            payoff: a function of S determining the payoff at the maturity.

        Returns:
            Option price in the binomial model.
        """
        n = np.arange(T + 1)
        p, q = self._risk_neutral_proba()
        V = np.zeros((T + 1, T + 1))
        S_T = S * (1 + self.u)**n * (1 + self.d)**(T - n)
        V[0] = payoff(S_T)
        for i in range(T):
            t = T - i - 1
            S_t = S * (1 + self.u)**n[:t + 1] * (1 + self.d)**(T - n[:t + 1])
            V[i + 1, :T - i] = np.maximum((p * V[i, 1:T + 1 - i] + q * V[i, :T - i]) / (1 + self.r), payoff(S_t))
        return V[-1, 0]

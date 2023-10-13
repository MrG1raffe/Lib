import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from typing import Union, Tuple
from numpy.typing import NDArray
from numpy import float_
from Diffusion import Diffusion

np.seterr(divide='ignore')


@dataclass
class Black76():
    sigma: Union[float, NDArray[float_]] = None
    r: float = 0
    correlation: NDArray[float_] = None
    dim: int = 1

    def _d1d2(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]]
    ) -> Tuple[Union[float, NDArray[float_]]]:
        """
        Calculates d1 and d2 from Black-76 formula.

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.

        Returns:
            d1: values of d1.
            d2: values of d2.
        """
        d1 = (np.log(F/K) + (self.sigma**2/2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2

    def vanilla_price(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the vanilla option price via Black-76 formula

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Prices of the call/put vanilla options.
        """
        d1, d2 = self._d1d2(T, K, F)
        if flag == 'c':
            return np.exp(-self.r*T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        if flag == 'p':
            return np.exp(-self.r*T) * (-F * norm.cdf(-d1) + K * norm.cdf(-d2))

    def digital_price(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the digital call or put option price via Black-76 formula

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Prices of the call/put vanilla options.
        """
        _, d2 = self._d1d2(T, K, F)

        sign = 1 if flag == 'c' else -1
        return np.exp(-self.r*T) * norm.cdf(sign * d2)

    def delta(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option delta in the Black-76 model

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, F)
        return norm.cdf(d1) if flag == 'c' else -norm.cdf(-d1)

    def gamma(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option gamma in the Black-76 model

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, F)
        return norm.pdf(d1) / (np.exp(-self.r * T) * F * self.sigma * np.sqrt(T))

    def vega(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
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
        d1, _ = self._d1d2(T, K, F)
        return np.exp(-self.r * T) * F * norm.pdf(d1) * np.sqrt(T)

    def simulate_trajectory(
        self,
        size: int,
        t_grid: Union[float, NDArray[float_]],
        init_val: Union[float, NDArray[float_]],
        flag: str = "forward",
        rng: np.random.Generator = None,
        antithetic_variates: bool = False,
        return_brownian: bool = False,
        return_diffusion: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Simulates the trajectory of stock or forward in the Black model.

        Args:
            size: number of simulated trajectories.
            t_grid: time grid to simulate the price on.
            init_val: the value of process at t = 0.
            flag: "forward" to simulate forward price (without drift). "spot" to simulate spot price.
            rng: `np.random.Generator` used for simulation.
            antithetic_variates: whether to use antithetic variantes in simulation. If True, the trajectories
                in the second half of the sample will use the random numbers opposite to the ones used in the first half.
            return_brownian: whether to return the underlying correlated brownian motion.
            return_diffusion: whether to return the corresponding diffusion object.

        Returns:
            np.ndarray of shape (size, len(t_grid)) with simulated trajectories if model dimension is 1.
            np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
        """
        if antithetic_variates:
            size = size // 2
        diffusion = Diffusion(
            t_grid=t_grid,
            size=size,
            dim=self.dim,
            rng=rng
        )
        drift = self.r if flag == "spot" else 0
        if antithetic_variates:
            diffusion.replace_brownian_motion(
                np.concatenate([
                    diffusion.brownian_motion(),
                    -diffusion.brownian_motion()
                ])
            )
        traj = diffusion.geometric_brownian_motion(
            init_val=init_val,
            drift=drift,
            correlation=self.correlation,
            vol=self.sigma,
            squeeze=True
        )
        W = diffusion.brownian_motion(correlation=self.correlation, squeeze=True)
        if return_diffusion:
            return traj, diffusion
        elif return_brownian:
            return traj, W
        else:
            return traj

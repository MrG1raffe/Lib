import math
import numpy as np
from typing import Union
from numpy.typing import NDArray
from numpy import float_
from utility import numpyze, is_number


def brownian_motion(
    n_sample: int,
    t_grid: Union[float, NDArray[float_]],
    init_val: Union[float, NDArray[float_]] = 0,
    drift: Union[float, NDArray[float_]] = 0,
    covariance: Union[float, NDArray[float_]] = 1,
    random_state: np.random.Generator = None,
    antithetic_variates: bool = False
) -> Union[float, NDArray[float_]]:
    """
    Simulates the trajectory of the d-dimensional Brownian motion.

    Args:
        n_sample: number of simulated trajectories.
        t_grid: time grid to simulate the price on.
        init_val: value of the process at t = 0.
        drift: number or vector of size d representing the constant drift.
        covariance: covariance matrix of the increments per unit time.
        random_state: `np.random.Generator` used for simulation.
        antithetic_variates: whether to use antithetic variantes in simulation. If True, the trajectories
            in the second half of the sample will use the random numbers opposite to the ones used in the first half.

    Returns:
        np.ndarray of shape (n_sample, len(t_grid)) with simulated trajectories if model dimension is 1.
        np.ndarray of shape (n_sample, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if random_state is None:
        random_state = np.random.default_rng()
    if antithetic_variates:
        n_sample_full = n_sample
        n_sample = math.ceil(n_sample / 2)
    dt = np.diff(np.concatenate([[0], t_grid]))
    t_grid = numpyze(t_grid)
    if is_number(covariance):
        dim = 1
    else:
        drift = numpyze(drift)
        init_val = numpyze(init_val)
        dim = len(covariance)
    if dim == 1:
        Z = random_state.normal(size=(n_sample, len(t_grid)))
        if antithetic_variates:
            Z = np.concatenate([Z, -Z[:n_sample_full - n_sample]], axis=0)
        traj = init_val + drift * t_grid + np.sqrt(covariance) * np.cumsum(np.sqrt(dt) * Z, axis=1)
    else:
        Z = random_state.multivariate_normal(
            mean=np.zeros(dim),
            cov=covariance,
            size=(n_sample, len(t_grid))
        )
        if antithetic_variates:
            Z = np.concatenate([Z, -Z[:n_sample_full - n_sample]], axis=0)
        traj = init_val[None, None, :] + drift[None, None, :] * t_grid[None, :, None] + \
            + np.cumsum(np.sqrt(dt)[None, :, None] * Z, axis=1)
        traj = traj.transpose([0, 2, 1])
    return traj


def geometric_brownian_motion(
    n_sample: int,
    t_grid: Union[float, NDArray[float_]],
    init_val: Union[float, NDArray[float_]] = 0,
    drift: Union[float, NDArray[float_]] = 0,
    covariance: Union[float, NDArray[float_]] = 1,
    random_state: np.random.Generator = None,
    antithetic_variates: bool = False
) -> Union[float, NDArray[float_]]:
    """
    Simulates the trajectory of the d-dimensional geometric Brownian motion.

    Args:
        n_sample: number of simulated trajectories.
        t_grid: time grid to simulate the price on.
        init_val: value of the process at t = 0.
        drift: number or vector of size d such that E[X_T] = exp(drift * T).
        covariance: covariance matrix of the log-increments per unit time.
        random_state: `np.random.Generator` used for simulation.
        antithetic_variates: whether to use antithetic variantes in simulation. If True, the trajectories
            in the first half of sample will be opposite to the ones in the second.

    Returns:
        np.ndarray of shape (n_sample, len(t_grid)) with simulated trajectories if model dimension is 1.
        np.ndarray of shape (n_sample, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if is_number(covariance):
        dim = 1
    else:
        dim = len(covariance)
    drift = drift - 0.5 * covariance if dim == 1 else drift - 0.5 * np.diag(covariance)
    W = brownian_motion(
        n_sample=n_sample,
        t_grid=t_grid,
        init_val=np.zeros(dim),
        drift=drift,
        covariance=covariance,
        random_state=random_state,
        antithetic_variates=antithetic_variates
    )
    if dim == 1:
        traj = init_val * np.exp(W)
    else:
        init_val = numpyze(init_val)
        traj = init_val[None, :, None] * np.exp(W)
    return traj

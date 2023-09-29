import numpy as np
from typing import Union, Callable
from numpy.typing import NDArray
from numpy import float_


def hedging_pnl(
    t_grid: NDArray[float_],
    S_traj: NDArray[float_],
    r: float,
    delta: Callable,
    init_value: Union[float, NDArray[float_]],
    payoff: Callable
) -> Union[float, NDArray[float_]]:
    """
    Performs hedging of the European derivative with given the time grid, trajectories of the underlyings, hedging strategy and the payoff.

    Args:
        t_grid: hedging time grid.
        S_traj: array of shape (n_sample, len(t_grid)) or len(t_grid) in case of one price trajectory.
        r: risk free rate. The price of zero coupon is assumed to be B(t, T) = exp(-r * (T - t)).
        delta: function of t and S_t returning the position in risky asset(s) at time t.
        init_value: value of the hedging portfolio at time t_grid[0].
        payoff: function of S, the payoff of the derivative at time 't_grid[-1]'.

    Returns:
        (Pnl, V): 'V' is the value of the hedging portfolio, i.e. an array of the same shape as 'S_traj' and 'Pnl' is a float
        or an array of length 'n_sample' with the P&L at time T = t_grid[-1] equal to V_T - payoff(S_T),
        where V_T is the terminal value of the hedging portfolio.
    """
    deltas = delta(t_grid, S_traj).T
    S_traj = S_traj.T
    cash = np.zeros_like(S_traj)
    V = np.zeros_like(S_traj)
    cash[0] = init_value - deltas[0] * S_traj[0]
    V[0] = init_value
    dt = np.diff(t_grid)
    d_deltas = np.diff(deltas, axis=0)
    for i in range(1, len(t_grid)):
        cash[i] = -d_deltas[i - 1] * S_traj[i] + np.exp(r * dt[i - 1]) * cash[i - 1]
    V = deltas * S_traj + cash
    return V[-1] - payoff(S_traj[-1]), V.T

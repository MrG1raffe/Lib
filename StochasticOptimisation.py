import numpy as np
from dataclasses import dataclass
from typing import Union, Callable
from numpy.typing import NDArray
from numpy import float_


@dataclass
class StochasticOptimisation:
    """
    Performs stochastic gradient algorithm

    theta[n + 1] = theta[n] - learning_rate(n + 1) * stoch_grad(theta[n], Z[0:batch_size]).

    Args:
        theta_init: initial value of theta, either constant or array.
        generate_gradient: a function of (batch_size, theta) that simulates the "gradient" or sample of "gradients" of size batch_size.
        learning_rate: a function of n or a constant number or array.
        batch_size: size of sample to compute the simulated gradient mean. If theta is array, sample elements should be indexed by axis 0.
        projector: function used to project theta on the constraint domain.
    """
    theta_init: Union[float, NDArray[float_]]
    generate_gradient: Callable[[int, Union[float, NDArray[float_]]], Union[float, NDArray[float_]]]
    learning_rate: Union[float, NDArray[float_], Callable[[int], Union[float, NDArray[float_]]]]
    batch_size: int = 1
    projector: Callable[[Union[float, NDArray[float_]]], Union[float, NDArray[float_]]] = lambda x: x
    iter_count: int = 0

    def step(
        self,
        n_steps: int,
        return_path: bool = False,
        reinit_theta: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Performs n_steps steps of the stochastic gradient algorithm. At the end reinitializes theta_init to the value of
        theta obtained at the last step if reinit_theta = True.

        Args:
            n_steps: number of steps to perform.
            return_path: whether to return an array of thetas instead of the last value.
            reinit_theta: whether to reinitialize theta and number of iterations at the end.

        Returns:
            theta: the last value of theta if return_path == False, an array of thetas otherwise.
        """
        lr = self.learning_rate if isinstance(self.learning_rate, Callable) else lambda n: self.learning_rate

        def get_step(i, theta):
            return lr(i + 1) * np.mean(self.generate_gradient(self.batch_size, theta), axis=0)

        if return_path:
            thetas = np.zeros((n_steps + 1,) + np.array(self.theta_init).shape)
            thetas[0] = self.theta_init
            for i in range(n_steps):
                thetas[i + 1] = self.projector(thetas[i] - get_step(self.iter_count + i, thetas[i]))
            if reinit_theta:
                self.set_theta(thetas[-1])
                self.iter_count += n_steps
            return thetas
        else:
            theta = self.theta_init
            for i in range(n_steps):
                theta = self.projector(theta - get_step(self.iter_count + i, theta))
            if reinit_theta:
                self.set_theta(theta)
                self.iter_count += n_steps
            return theta

    def set_theta(
        self,
        theta: Union[float, NDArray[float_]]
    ) -> None:
        """
        Reinitializes theta_init.
        """
        self.theta_init = theta
        return

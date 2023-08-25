import numpy as np
import warnings
from scipy.optimize import minimize, LinearConstraint


def portfolio_return(t_w: np.ndarray, t_mu: np.ndarray) -> float:
    return t_w.dot(t_mu)


def portfolio_variance(t_w: np.ndarray, t_sigma: np.ndarray) -> float:
    return t_w.dot(t_sigma).dot(t_w)


def portfolio_utility(t_w: np.ndarray, t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float) -> float:
    # u = -2 w*m/l +wSw  <=> v = w*m - l/2 * wSw
    return -2 * portfolio_return(t_w, t_mu) / t_lbd + portfolio_variance(t_w, t_sigma)


def minimize_utility_con(t_mu: np.ndarray, t_sigma: np.ndarray, t_lbd: float,
                         t_bounds: tuple = (0, 1), t_pos_lim: tuple = (0, 1),
                         t_tol: float = None) -> (np.ndarray, float):
    _p, _ = t_sigma.shape
    warnings.filterwarnings("ignore")
    _res = minimize(
        fun=portfolio_utility,
        x0=np.ones(_p) / _p,
        args=(t_mu, t_sigma, t_lbd),
        bounds=[t_bounds] * _p,
        # constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
        # constraints= scipy.optimize.LinearConstraint(np.ones(_p), -1, 1)
        constraints=LinearConstraint(np.ones(_p), t_pos_lim[0], t_pos_lim[1]),
        tol=t_tol,
    )
    warnings.filterwarnings("always")
    if _res.success:
        return _res.x, _res.fun
    else:
        print("ERROR! Optimizer exits with a failure")
        print("Detailed Description: {}".format(_res.message))
        return None, None

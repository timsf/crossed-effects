from typing import Callable, Tuple

import numpy as np


def sample_marginal(x: np.ndarray, mu: np.ndarray, u: np.ndarray, l_tau: np.ndarray, delt: np.ndarray,
                    f_log_f: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                    ome: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:

    l_b = l_tau[np.newaxis] + 1 / delt[:, np.newaxis]
    l_a = 1 / l_b
    l_cov = l_a + np.square(l_a) / delt[:, np.newaxis]
    l_prec = 1 / l_cov
    tau = (u * l_tau) @ u.T

    x_log_f, dx_log_f = f_log_f(x)
    x_log_p = x_log_f + eval_norm_prec(x, mu, u, l_tau[np.newaxis])
    mean_x = (((x / delt[:, np.newaxis] + mu @ tau + dx_log_f) @ u) * l_a) @ u.T
    y = sample_norm_cov(mean_x, u, l_cov, ome)

    y_log_f, dy_log_f = f_log_f(y)
    y_log_p = y_log_f + eval_norm_prec(y, mu, u, l_tau[np.newaxis])
    mean_y = (((y / delt[:, np.newaxis] + mu @ tau + dy_log_f) @ u) * l_a) @ u.T

    log_post_odds = y_log_p - x_log_p
    log_prop_odds = eval_norm_prec(y, mean_x, u, l_prec) - eval_norm_prec(x, mean_y, u, l_prec)
    acc_prob = np.exp([min(0, lp) for lp in log_post_odds - log_prop_odds])

    return np.where(ome.uniform(size=x.shape[0]) < acc_prob, y.T, x.T).T, acc_prob


def eval_norm_prec(x: np.ndarray, mu: np.ndarray, u: np.ndarray, l_tau: np.ndarray) -> np.ndarray:

    mah = np.sum(np.square(((x - mu) @ u) * np.sqrt(l_tau)), 1)
    return (np.sum(np.log(l_tau), 1) - mah - x.shape[1] * np.log(2 * np.pi)) / 2


def sample_norm_cov(mu: np.ndarray, u: np.ndarray, l_sig: np.ndarray, ome: np.random.Generator) -> np.ndarray:

    z = ome.standard_normal(mu.shape)
    return mu + (z * np.sqrt(l_sig)) @ u.T


class LatentGaussSampler(object):

    def __init__(self, n: np.array, opt_prob: float = .5):

        self.n = n
        self.emp_prob = [np.ones(len(n))]
        self.step = [-np.log(n)]
        self.opt_prob = opt_prob

    def sample(self, x_nil: np.ndarray, mu: np.ndarray, u: np.ndarray, l_tau: np.ndarray,
               f_log_f: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], ome: np.random.Generator) -> np.ndarray:

        x_prime, emp_prob = sample_marginal(x_nil, mu, u, l_tau, np.exp(self.step[-1]), f_log_f, ome)
        self.emp_prob.append(emp_prob)
        self.step.append(self.step[-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob)))
        return x_prime

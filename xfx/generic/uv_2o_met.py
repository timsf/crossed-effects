from typing import Callable, Tuple

import numpy as np


def sample(x: np.ndarray, mu: np.ndarray, tau: np.ndarray,
           f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], ome: np.random.Generator
           ) -> Tuple[np.ndarray, np.ndarray]:

    x_log_p, mean_x, prec_x = ascend(x, mu, tau, f_log_p)
    y = ome.normal(mean_x, 1 / np.sqrt(prec_x))
    y_log_p, mean_y, prec_y = ascend(y, mu, tau, f_log_p)
    return accept_reject(x, y, x_log_p, y_log_p, mean_x, mean_y, prec_x, prec_y, mu, tau, ome)


def ascend(x: np.ndarray, mu: np.ndarray, tau: np.ndarray,
                  f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_log_p, dx_log_p, d2x_log_p = f_log_p(x)
    x_hess = tau - d2x_log_p
    x_prime = (dx_log_p + tau * mu - d2x_log_p * x) / x_hess
    return x_log_p, x_prime, x_hess


def accept_reject(x: np.ndarray, y: np.ndarray, x_log_p: np.ndarray, y_log_p: np.ndarray,
                  mean_x: np.ndarray, mean_y: np.ndarray, prec_x: np.ndarray, prec_y: np.ndarray,
                  mu: np.ndarray, tau: np.ndarray, ome: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:

    log_lik_ratio = y_log_p - x_log_p
    log_prior_odds = eval_norm(y, mu, tau) - eval_norm(x, mu, tau)
    log_prop_odds = eval_norm(y, mean_x, prec_x) - eval_norm(x, mean_y, prec_y)
    log_acc_odds = log_lik_ratio + log_prior_odds - log_prop_odds
    acc_prob = np.exp([min(0, lp) for lp in np.where(np.isnan(log_acc_odds), -np.inf, log_acc_odds)])
    return np.where(ome.uniform(size=len(x)) < acc_prob, y, x), acc_prob


def eval_norm(x: np.ndarray, mu: np.ndarray, tau: np.ndarray) -> np.ndarray:

    d = (x - mu) ** 2 * tau
    kern = -d / 2
    cons = (np.log(tau) - np.log(2 * np.pi)) / 2
    return cons + kern


class LatentGaussSampler(object):

    def __init__(self, j: int):

        self.emp_prob = [np.ones(j)]

    def sample(self, x_nil: np.ndarray, mu: np.ndarray, tau: np.ndarray,
               f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], ome: np.random.Generator
               ) -> np.ndarray:

        x_prime, acc_prob = sample(x_nil, mu, tau, f_log_p, ome)
        self.emp_prob.append(acc_prob)
        return x_prime

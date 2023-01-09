from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def sample_marginal(
    x: FloatArr,
    mu: FloatArr,
    u: FloatArr,
    l_tau: FloatArr,
    delt: FloatArr,
    f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr]],
    ome: np.random.Generator,
) -> Tuple[FloatArr, FloatArr]:

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
    log_acc_odds = log_post_odds - log_prop_odds
    acc_prob = np.exp([min(0, lp) for lp in np.where(np.isnan(log_acc_odds), -np.inf, log_acc_odds)])

    return np.where(ome.uniform(size=x.shape[0]) < acc_prob, y.T, x.T).T, acc_prob


def eval_norm_prec(
    x: FloatArr,
    mu: FloatArr,
    u: FloatArr,
    l_tau: FloatArr,
) -> FloatArr:

    mah = np.sum(np.square(((x - mu) @ u) * np.sqrt(l_tau)), 1)
    return (np.sum(np.log(l_tau), 1) - mah - x.shape[1] * np.log(2 * np.pi)) / 2


def sample_norm_cov(
    mu: FloatArr,
    u: FloatArr,
    l_sig: FloatArr,
    ome: np.random.Generator,
) -> FloatArr:

    z = ome.standard_normal(mu.shape)
    return mu + (z * np.sqrt(l_sig)) @ u.T


class LatentGaussSampler(object):

    def __init__(self, j: int, opt_prob: float = .5):

        self.emp_prob = [np.ones(j)]
        self.step = [-np.zeros(j)]
        self.opt_prob = opt_prob

    def sample(
        self,
        x_nil: FloatArr,
        mu: FloatArr,
        u: FloatArr,
        l_tau: FloatArr,
        f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr]],
        ome: np.random.Generator,
    ) -> FloatArr:

        x_prime, emp_prob = sample_marginal(x_nil, mu, u, l_tau, np.exp(self.step[-1]), f_log_f, ome)
        self.emp_prob.append(emp_prob)
        self.step.append(self.step[-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob)))
        return x_prime

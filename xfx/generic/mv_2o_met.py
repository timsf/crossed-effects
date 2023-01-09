from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def sample_marginal(
    x: FloatArr,
    mu: FloatArr,
    tau: FloatArr,
    delt: FloatArr,
    f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]],
    ome: np.random.Generator,
) -> Tuple[FloatArr, FloatArr]:

    l_tau, u = np.linalg.eigh(tau)
    x_log_f, dx_log_f, d2x_log_f = f_log_f(x)
    tau_x = tau - d2x_log_f
    l_tau_x, u_x = np.linalg.eigh(tau_x)
    l_b_x = l_tau_x[np.newaxis] + 1 / delt[:, np.newaxis]
    l_a_x = 1 / l_b_x
    l_cov_x = l_a_x + np.square(l_a_x) / delt[:, np.newaxis]
    l_prec_x = 1 / l_cov_x

    x_log_p = x_log_f + eval_norm_prec(x, mu, u, l_tau[np.newaxis])
    mean_x = (((x / delt[:, np.newaxis] - x @ d2x_log_f + mu @ tau + dx_log_f) @ u_x) * l_a_x) @ u_x.T
    y = sample_norm_cov(mean_x, u_x, l_cov_x, ome)

    y_log_f, dy_log_f, d2y_log_f = f_log_f(y)
    tau_y = tau - d2y_log_f
    l_tau_y, u_y = np.linalg.eigh(tau_y)
    l_b_y = l_tau_y[np.newaxis] + 1 / delt[:, np.newaxis]
    l_a_y = 1 / l_b_y
    l_cov_y = l_a_y + np.square(l_a_y) / delt[:, np.newaxis]
    l_prec_y = 1 / l_cov_y

    y_log_p = y_log_f + eval_norm_prec(y, mu, u, l_tau[np.newaxis])
    mean_y = (((y / delt[:, np.newaxis] - y @ d2y_log_f + mu @ tau + dy_log_f) @ u_y) * l_a_y) @ u_y.T

    log_post_odds = y_log_p - x_log_p
    log_prop_odds = eval_norm_prec(y, mean_x, u_x, l_prec_x) - eval_norm_prec(x, mean_y, u_y, l_prec_y)
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
        self.step = [np.zeros(j)]
        self.opt_prob = opt_prob

    def sample(
        self,
        x_nil: FloatArr,
        mu: FloatArr,
        tau: FloatArr,
        f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]],
        ome: np.random.Generator,
    ) -> FloatArr:

        try:
            x_prime, emp_prob = sample_marginal(x_nil, mu, tau, np.exp(self.step[-1]), f_log_f, ome)
        except np.linalg.LinAlgError:
            x_prime, emp_prob = x_nil, np.zeros(x_nil.shape[0])
        self.emp_prob.append(emp_prob)
        self.step.append(self.step[-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob)))
        return x_prime

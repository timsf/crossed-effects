from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

import xfx.generic.mv_conjugate
from xfx.generic.mv_1o_met import LatentGaussSampler


PartFunc = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def sample_posterior(y: np.ndarray, n: np.ndarray, j: np.ndarray, i: np.ndarray,
                     eval_part: PartFunc, tau0: Optional[np.ndarray],
                     prior_n_tau: Optional[np.ndarray], prior_est_tau: Optional[List[np.ndarray]],
                     init: Optional[Tuple[List[np.ndarray], List[np.ndarray]]], 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:

    if tau0 is None:
        tau0 = np.identity(y.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(y.shape[1], len(j))
    if prior_est_tau is None:
        prior_est_tau = len(j) * [np.identity(y.shape[1])]

    if init is None:
        alp0 = np.zeros(y.shape[1])
        alp = [np.zeros((j_, y.shape[1])) for j_ in j]
        tau = prior_est_tau
    else:
        alp0[0][0], alp[1:], tau = init

    i_ord = np.argsort(i, 0)
    samplers = [LatentGaussSampler(n) for n in [np.bincount(i_) for i_ in i.T]]

    while True:
        alp0, alp = update_coefs(y, n, i, i_ord, alp0, alp, tau0, tau, eval_part, samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.mv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        yield [alp0[np.newaxis]] + alp, tau


def update_coefs(y: np.ndarray, n: np.ndarray, i: np.ndarray, i_ord: np.ndarray,
                 alp0: np.ndarray, alp: List[np.ndarray], tau0: np.ndarray, tau: List[np.ndarray],
                 eval_part: PartFunc, samplers: List[LatentGaussSampler], ome: np.random.Generator
                 ) -> Tuple[float, List[np.ndarray]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y, n, i, i_ord, k_, new_alp0, new_alp, tau0, tau_, eval_part,
                                                   sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(y: np.ndarray, n: np.ndarray, i: np.ndarray, i_ord: Tuple[int, np.ndarray], k_: int,
                       alp0: np.ndarray, alp: List[np.ndarray], tau0: np.ndarray, tau_: np.ndarray,
                       eval_part: PartFunc, sampler: LatentGaussSampler, ome: np.random.Generator
                       ) -> Tuple[float, np.ndarray]:

    def eval_log_f(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        log_p, dk_log_p = eval_kernel(y, n, i, i_ord, alp0, alp[:k_] + [b - alp0] + alp[(k_ + 1):], eval_part, k_)
        return log_p, dk_log_p

    l_tau, u = np.linalg.eigh(tau_)
    new_bet_ = sampler.sample(alp[k_] + alp0, alp0, u, l_tau, eval_log_f, ome)
    prec_alp0 = tau0 + alp[k_].shape[0] * tau_
    mean_alp0 = np.linalg.solve(prec_alp0, tau_ @ np.sum(new_bet_, 0))
    new_alp0 = mean_alp0 + np.linalg.solve(np.linalg.cholesky(prec_alp0), ome.standard_normal(alp[k_].shape[1]))
    return new_alp0, new_bet_ - new_alp0


def eval_kernel(y: np.ndarray, n: np.ndarray, i: np.ndarray, i_ord: np.ndarray, alp0: np.ndarray, alp: List[np.ndarray],
                eval_part: PartFunc, k_: int = None) -> Tuple[np.ndarray, np.ndarray]:

    eta = alp0 + sum([alp_[i_] for alp_, i_ in zip(alp, i.T)])
    part, d_part = eval_part(eta)

    log_f = np.sum(y * eta, 1) - n * part
    d_log_f = y - n[:, np.newaxis] * d_part

    brk = np.cumsum(np.bincount(i[:, k_]))[:-1]
    return tuple([groupby(dn_log_f, i_ord[:, k_], brk, np.sum) for dn_log_f in (log_f, d_log_f)])


def groupby(arr: np.ndarray, ord: np.ndarray, brk: np.ndarray, f: Callable[[np.ndarray, int], float]) -> np.ndarray:

    return np.array([f(a, axis=0) for a in np.split(arr[ord], brk)])


# def update_single_coef(ik: np.ndarray, yk: GlmSuffStat, block: (int, np.ndarray),
#                        alp0: np.ndarray, alp: [np.ndarray], tau0: float, tau_: np.ndarray,
#                        eval_part: PartFunc, sampler: LatentGaussSampler) -> (float, np.ndarray):
#
#     def eval_log_f(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         log_p, dk_log_p = eval_loglik(ik, yk, alp0, alp[:k_] + [b - alp0] + alp[(k_ + 1):], eval_part, block)
#         return log_p, dk_log_p
#
#     k_, _ = block
#     l_tau_, u_ = np.linalg.eigh(tau_)
#     new_bet_ = sampler.sample(alp[k_] + alp0, alp0, u_, l_tau_, eval_log_f)
#
#     mean_alp0 = np.linalg.solve(tau0 * np.identity(alp[k_].shape[1]) + alp[k_].shape[0] * tau_, tau_ @ np.sum(new_bet_, 0))
#     new_alp0, = sample_norm_cov(mean_alp0, u_, (1 / (tau0 + alp[k_].shape[0] * l_tau_))[np.newaxis])
#
#     return new_alp0, new_bet_ - new_alp0

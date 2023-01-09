from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import xfx.generic.mv_conjugate
from xfx.generic.mv_1o_met import LatentGaussSampler


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]
PartFunc = Callable[[FloatArr], Tuple[FloatArr, FloatArr]]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    eval_part: PartFunc,
    tau0: Optional[FloatArr],
    prior_n_tau: Optional[FloatArr],
    prior_est_tau: Optional[List[FloatArr]],
    init: Optional[Tuple[List[FloatArr], List[FloatArr]]],
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[Tuple[List[FloatArr], List[FloatArr]]]:

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
        alp, tau = init
        alp0, alp = alp[0][0], alp[1:]

    i_ord = np.argsort(i, 0)
    sampler0 = LatentGaussSampler(1)
    samplers = [LatentGaussSampler(j_) for j_ in j]

    while True:
        alp0, alp = update_coefs(y, n, j, i, i_ord, alp0, alp, tau0, tau, eval_part, sampler0, samplers, collapse, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.mv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        yield [alp0[np.newaxis]] + alp, tau


def update_coefs(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: FloatArr,
    alp: List[FloatArr],
    tau0: FloatArr,
    tau: List[FloatArr],
    eval_part: PartFunc,
    sampler0: LatentGaussSampler,
    samplers: List[LatentGaussSampler],
    collapse: bool,
    ome: np.random.Generator,
) -> Tuple[FloatArr, List[FloatArr]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y, n, j, i, i_ord, k_, new_alp0, new_alp, tau0, tau_, eval_part,
                                                   sampler0, sampler_, collapse, ome)
    return new_alp0, new_alp


def update_single_coef(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    k_: int,
    alp0: FloatArr,
    alp: List[FloatArr],
    tau0: FloatArr,
    tau_: FloatArr,
    eval_part: PartFunc,
    sampler0: LatentGaussSampler,
    sampler: LatentGaussSampler,
    collapse: bool,
    ome: np.random.Generator,
) -> Tuple[FloatArr, FloatArr]:

    def eval_log_f(b: FloatArr) -> Tuple[FloatArr, FloatArr]:
        log_p, dk_log_p = eval_kernel(y, n, j, i, i_ord, alp0, alp[:k_] + [b - alp0] + alp[(k_ + 1):], eval_part, k_)
        return log_p, dk_log_p

    l_tau, u = np.linalg.eigh(tau_)
    new_bet_ = sampler.sample(alp[k_] + alp0, alp0, u, l_tau, eval_log_f, ome)
    if collapse:
        prec_alp0 = tau0 + alp[k_].shape[0] * tau_
        mean_alp0 = np.linalg.solve(prec_alp0, tau_ @ np.sum(new_bet_, 0))
        new_alp0 = mean_alp0 + np.linalg.solve(np.linalg.cholesky(prec_alp0), ome.standard_normal(alp[k_].shape[1]))
        new_alp_ = new_bet_ - new_alp0
    else:
        new_alp_ = new_bet_ - alp0
        new_alp0 = update_intercept(y, n, j, i, i_ord, alp0, alp[:k_] + [new_alp_] + alp[(k_ + 1):], tau0, eval_part, sampler0, ome)

    return new_alp0, new_alp_


def update_intercept(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: FloatArr,
    alp: List[FloatArr],
    tau0: FloatArr,
    eval_part: PartFunc,
    sampler0: LatentGaussSampler,
    ome: np.random.Generator,
) -> FloatArr:

    def eval_log_f(a: FloatArr) -> Tuple[FloatArr, FloatArr]:
        log_p, dk_log_p = eval_kernel(y, n, j, i, i_ord, a[0], alp, eval_part, None)
        return log_p, dk_log_p

    l_tau, u = np.linalg.eigh(tau0)
    new_alp0 = sampler0.sample(alp0[np.newaxis], np.zeros_like(alp0), u, l_tau, eval_log_f, ome)[0]
    return new_alp0


def eval_kernel(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: FloatArr,
    alp: List[FloatArr],
    eval_part: PartFunc,
    k_: int = None,
) -> Tuple[FloatArr, FloatArr]:

    eta = alp0 + sum([alp_[i_] for alp_, i_ in zip(alp, i.T)])
    part, d_part = eval_part(eta)

    log_f = np.sum(y * eta, 1) - n * part
    d_log_f = y - n[:, np.newaxis] * d_part

    if k_ is not None:
        brk = np.cumsum(np.bincount(i[:, k_], minlength=j[k_]))[:-1]
        return groupby(log_f, i_ord[:, k_], brk), groupby(d_log_f, i_ord[:, k_], brk)
    return np.sum(log_f, 0)[np.newaxis], np.sum(d_log_f, 0)[np.newaxis]


def groupby(
    arr: FloatArr,
    ord: FloatArr,
    brk: FloatArr,
) -> FloatArr:

    return np.float_([np.sum(a, axis=0) for a in np.split(arr[ord], brk)])

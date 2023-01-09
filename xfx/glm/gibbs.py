from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar

import xfx.generic.uv_conjugate
from xfx.generic.uv_2o_met import LatentGaussSampler


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]
PartFunc = Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]]
BaseFunc = Callable[[FloatArr, FloatArr, FloatArr, float], Tuple[float, float, float]]


def sample_disp_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    eval_part: PartFunc,
    eval_base: BaseFunc,
    prior_n_tau: Optional[FloatArr],
    prior_est_tau: Optional[FloatArr],
    prior_n_phi: Optional[float],
    prior_est_phi: Optional[float],
    init: Optional[Tuple[List[FloatArr], FloatArr, float]],
    collapse: bool,
    ome: np.random.Generator,
) -> Iterator[Tuple[List[FloatArr], FloatArr, float]]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))
    if prior_n_phi is None:
        prior_n_phi = 1
    if prior_est_phi is None:
        prior_est_phi = 1

    if init is None:
        alp0 = 0
        alp = [np.zeros(j_) for j_ in j]
        tau = prior_est_tau
        phi = 1
    else:
        alp, tau, phi = init
        alp0, alp = alp[0][0], alp[1:]

    i_ord = np.argsort(i, 0)
    samplers = [LatentGaussSampler(j_) for j_ in j]

    while True:
        alp0, alp = update_coefs(y1, n, j, i, i_ord, alp0, alp, tau, phi, collapse, eval_part, samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        if not np.isinf(prior_n_phi):
            phi = update_dispersion(y1, y2, n, j, i, i_ord, alp0, alp, phi,
                                    eval_part, eval_base, prior_n_phi, prior_est_phi, ome)
        yield [np.array([alp0])] + alp, tau, phi


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    eval_part: PartFunc,
    prior_n_tau: Optional[FloatArr],
    prior_est_tau: Optional[FloatArr],
    init: Optional[Tuple[List[FloatArr], FloatArr]],
    collapse: bool,
    ome: np.random.Generator,
) -> Iterator[Tuple[List[FloatArr], FloatArr]]:

    eval_base = lambda _, __, ___, ____: (0, 0, 0)
    return (the[:-1] for the in
            sample_disp_posterior(y, np.empty_like(y), n, j, i, eval_part, eval_base, prior_n_tau, prior_est_tau, np.inf, 1,
                                  init if init is None else init + (1,), collapse, ome))


def update_coefs(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: List[FloatArr],
    tau: FloatArr,
    phi: float,
    collapse: bool,
    eval_part: PartFunc,
    samplers: List[LatentGaussSampler],
    ome: np.random.Generator,
) -> Tuple[float, List[FloatArr]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y1, n, j, i, i_ord, k_, new_alp0, new_alp, tau_,
                                                   phi, collapse, eval_part, sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    k_: int,
    alp0: float,
    alp: List[FloatArr],
    tau_: float,
    phi: float,
    collapse: bool,
    eval_part: PartFunc,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> Tuple[float, FloatArr]:

    def eval_log_p(b: FloatArr) -> Tuple[FloatArr, FloatArr, FloatArr]:
        log_p, dk_log_p, d2k_log_p = eval_kernel(y1, n, j, i, i_ord, alp0, alp[:k_] + [b - alp0] + alp[(k_ + 1):],
                                                 eval_part, k_)
        return log_p / phi, dk_log_p / phi, d2k_log_p / phi

    new_bet_ = sampler.sample(alp[k_] + alp0, np.repeat(alp0, len(alp[k_])), np.repeat(tau_, len(alp[k_])),
                              eval_log_p, ome)
    if collapse:
        new_alp0 = ome.normal(np.mean(new_bet_), 1 / np.sqrt(tau_ * len(alp[k_])))
        new_alp_ = new_bet_ - new_alp0
    else:
        new_alp_ = new_bet_ - alp0
        new_alp0 = update_intercept(y1, n, j, i, i_ord, alp0, alp[:k_] + [new_alp_] + alp[(k_ + 1):],
                                    0, phi, eval_part, ome)
    return new_alp0, new_alp_


def update_intercept(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: List[FloatArr],
    tau0: float,
    phi: float,
    eval_part: PartFunc,
    ome: np.random.Generator,
) -> float:

    def eval_log_p(a: FloatArr) -> Tuple[FloatArr, FloatArr, FloatArr]:
        log_p, dk_log_p, d2k_log_p = eval_kernel(y1, n, j, i, i_ord, a[0], alp, eval_part, None)
        return log_p / phi, dk_log_p / phi, d2k_log_p / phi

    sampler = LatentGaussSampler(1)
    return sampler.sample(np.float_([alp0]), np.zeros(1), np.float_([tau0 if tau0 != 0 else np.finfo(float).eps]),
                          eval_log_p, ome)[0]


def update_dispersion(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: List[FloatArr],
    phi: float,
    eval_part: PartFunc,
    eval_base: BaseFunc,
    prior_n: float,
    prior_est: float,
    ome: np.random.Generator,
) -> float:

    def eval_log_p(phi_: float, log_v: float) -> Tuple[float, float, float]:
        log_g, d_log_g, d2_log_g = eval_base(y1, y2, n, phi_)
        log_prior, d_log_prior, d2_log_prior = eval_logprior_phi(phi_, prior_n, prior_est)
        log_p = log_prior + log_g + log_p_nil / phi_ - log_v
        d_log_p = d_log_prior + d_log_g - log_p_nil / phi_ ** 2
        d2_log_p = d2_log_prior + d2_log_g + 2 * log_p_nil / phi_ ** 3
        return log_p, d_log_p, d2_log_p

    def brace(right: bool) -> float:
        sgn = 1 if right else -1
        width = 1
        while True:
            edge = phi * 2 ** (sgn * width)
            log_p, d_log_p, _ = eval_log_p(edge, log_u)
            if log_p < 0 and sgn * d_log_p < 0:
                return edge
            width += 1

    log_p_nil, = eval_kernel(y1, n, j, i, i_ord, alp0, alp, eval_part)[0]
    log_u = eval_log_p(phi, 0)[0] - ome.exponential()
    lb = root_scalar(eval_log_p, (log_u,), bracket=(brace(False), phi), fprime=True, fprime2=True).root
    ub = root_scalar(eval_log_p, (log_u,), bracket=(phi, brace(True)), fprime=True, fprime2=True).root
    return ome.uniform(lb, ub)


def eval_logprior_phi(phi: float, prior_n: float, prior_est: float) -> Tuple[float, float, float]:

    return -(prior_n / 2 + 1) * np.log(phi) - prior_n * prior_est / (2 * phi), \
           -(prior_n / 2 + 1) / phi + prior_n * prior_est / (2 * phi ** 2), \
           (prior_n / 2 + 1) / phi ** 2 - prior_n * prior_est / phi ** 3


def eval_kernel(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    alp0: float,
    alp: List[FloatArr],
    eval_part: PartFunc,
    k_: int = None
) -> Tuple[FloatArr, FloatArr, FloatArr]:

    eta = alp0 + np.sum(np.float_([alp_[i_] for alp_, i_ in zip(alp, i.T)]), 0)
    part, d_part, d2_part = eval_part(eta)

    log_f = y1 * eta - n * part
    d_log_f = y1 - n * d_part
    d2_log_f = - n * d2_part

    if k_ is not None:
        brk = np.cumsum(np.bincount(i[:, k_], minlength=j[k_]))[:-1]
        return groupby(log_f, i_ord[:, k_], brk), groupby(d_log_f, i_ord[:, k_], brk), \
               groupby(d2_log_f, i_ord[:, k_], brk)
    return np.sum(log_f, 0)[np.newaxis], np.sum(d_log_f, 0)[np.newaxis], np.sum(d2_log_f, 0)[np.newaxis]


def groupby(arr: FloatArr, ord: FloatArr, brk: FloatArr) -> FloatArr:

    return np.float_([np.sum(a) for a in np.split(arr[ord], brk)])

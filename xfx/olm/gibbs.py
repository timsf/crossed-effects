from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

import xfx.generic.uv_conjugate
from xfx.generic.uv_2o_met import LatentGaussSampler as UvLatentGaussSampler
from xfx.generic.mv_2o_met import LatentGaussSampler as MvLatentGaussSampler


Cdfunc = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def sample_posterior(y: np.ndarray, j: np.ndarray, i: np.ndarray, eval_cdf: Cdfunc,
                     prior_n_tau: Optional[np.ndarray], prior_est_tau: Optional[np.ndarray],
                     prior_n_ups: Optional[float],
                     init: Optional[Tuple[List[np.ndarray], np.ndarray, np.ndarray]],
                     ome: np.random.Generator
                     ) -> Iterator[Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))

    if init is None:
        alp0 = 0
        alp = [np.zeros(j_) for j_ in j]
        tau = prior_est_tau
        ups = np.arange(max(y))
    else:
        alp0, alp, tau, ups = init
    
    i_ord = np.argsort(i, 0)
    l_ord = np.argsort(y, 0)
    alp_samplers = [UvLatentGaussSampler(n) for n in [np.bincount(i_) for i_ in i.T]]
    ups_sampler = MvLatentGaussSampler([max(y)])

    while True:
        alp0, alp = update_coefs(y, i, i_ord, alp0, alp, tau, ups, eval_cdf, alp_samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        alp0, ups = update_thresholds(y, i, l_ord, alp0, alp, ups, prior_n_ups, eval_cdf, ups_sampler, ome)
        yield [np.array([alp0])] + alp, tau, ups


def update_coefs(y: np.ndarray, i: np.ndarray, i_ord: np.ndarray,
                 alp0: float, alp: List[np.ndarray], tau: np.ndarray, ups: np.ndarray,
                 eval_cdf: Cdfunc, samplers: List[UvLatentGaussSampler], ome: np.random.Generator
                 ) -> Tuple[float, List[np.ndarray]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y, i, i_ord, k_, new_alp0, new_alp, tau_, ups,
                                                   eval_cdf, sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(y: np.ndarray, i: np.ndarray, i_ord: np.ndarray, k_: int,
                       alp0: float, alp: List[np.ndarray], tau_: float, ups: np.ndarray, eval_cdf: Cdfunc,
                       sampler: UvLatentGaussSampler, ome: np.random.Generator
                       ) -> Tuple[float, np.ndarray]:

    def eval_log_p(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_p, dk_log_p, d2k_log_p = eval_coef_blocks(y, i, i_ord, alp0, alp[:k_] + [b - alp0] + alp[(k_ + 1):], ups, eval_cdf, k_)
        return log_p, dk_log_p, d2k_log_p

    new_bet_ = sampler.sample(alp[k_] + alp0, np.repeat(alp0, len(alp[k_])), np.repeat(tau_, len(alp[k_])), eval_log_p, ome)
    new_alp0 = ome.normal(np.mean(new_bet_), 1 / np.sqrt(tau_ * len(alp[k_])))
    new_alp_ = new_bet_ - new_alp0
    return new_alp0, new_alp_


def eval_coef_blocks(y: np.ndarray, i: np.ndarray, i_ord: np.ndarray, alp0: float, alp: List[np.ndarray], ups: np.ndarray, 
                     eval_cdf: Cdfunc, k_: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ups_ext = np.hstack([-np.inf, ups, np.inf])[np.vstack([y, y+1]).T]
    eta = ups_ext - alp0 - sum([alp_[i_] for alp_, i_ in zip(alp, i.T)])[:, np.newaxis]

    cdf, deta_cdf, d2eta_cdf = eval_cdf(eta)
    pmf, deta_pmf, d2eta_pmf = (np.diff(cdf_, 1)[:, 0] for cdf_ in (cdf, deta_cdf, d2eta_cdf))

    log_f = np.log(pmf)
    d_log_f = -deta_pmf / pmf
    d2_log_f = d2eta_pmf / pmf - np.square(deta_pmf / pmf)

    if k_ is not None:
        brk = np.cumsum(np.bincount(i[:, k_]))[:-1]
        return tuple([groupby(dn_log_f, i_ord[:, k_], brk, np.sum) for dn_log_f in (log_f, d_log_f, d2_log_f)])
    return tuple([np.sum(dn_log_f, 0)[np.newaxis] for dn_log_f in (log_f, d_log_f, d2_log_f)])


def update_thresholds(y: np.ndarray, i: np.ndarray, l_ord: np.ndarray,
                      alp0: float, alp: List[np.ndarray], ups: np.ndarray, prior_n_ups: float,
                      eval_cdf: Cdfunc, sampler: MvLatentGaussSampler, ome: np.random.Generator
                      ) -> Tuple[float, np.ndarray]:

    def eval_log_p(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.any(np.diff(p[0]) < 0):
            return np.array([-np.inf]), np.zeros((1, len(p[0]))), np.zeros(2 * (len(p[0]),))
        log_p, d_log_p, d2_log_p = eval_thresh_blocks(y, i, l_ord, alp, p[0], eval_cdf)
        return np.array([log_p]), d_log_p[np.newaxis], d2_log_p

    phi = ups - alp0
    prec_phi = prior_n_ups * np.linalg.inv(np.min(np.mgrid[:len(phi), :len(phi)], 0) + 1)
    new_phi = sampler.sample(phi[np.newaxis], np.repeat(-alp0, len(phi)), prec_phi, eval_log_p, ome)[0]
    new_alp0 = ome.normal(-new_phi[0], 1 / np.sqrt(prior_n_ups))
    new_ups = new_phi + new_alp0
    return new_alp0, new_ups


def eval_thresh_blocks(y: np.ndarray, i: np.ndarray, l_ord: np.ndarray, alp: List[np.ndarray], phi: np.ndarray, 
                       eval_cdf: Cdfunc) -> Tuple[float, np.ndarray, np.ndarray]:

    phi_ext = np.hstack([-np.inf, phi, np.inf])[np.vstack([y, y+1]).T]
    eta = phi_ext - sum([alp_[i_] for alp_, i_ in zip(alp, i.T)])[:, np.newaxis]

    cdf, deta_cdf, d2eta_cdf = eval_cdf(eta)
    pmf = np.diff(cdf, 1)[:, 0]

    log_f = np.log(pmf)
    d_log_f = (deta_cdf.T / pmf).T
    d2_log_f = np.array([np.diag(d2eta_cdf_ / pmf_) - np.outer(deta_cdf_, deta_cdf_) / np.square(pmf_) for pmf_, deta_cdf_, d2eta_cdf_ in zip(pmf, deta_cdf, d2eta_cdf)])

    brk = np.cumsum(np.bincount(y))[:-1]
    grp_log_f, grp_d_log_f, grp_d2_log_f = tuple([groupby(dn_log_f, l_ord, brk, sum) for dn_log_f in (log_f, d_log_f, d2_log_f)])

    log_p = np.sum(grp_log_f)
    d_log_p = np.zeros(len(phi))
    d_log_p[0] += grp_d_log_f[0, -1]
    for y_ in range(1, max(y)):
        d_log_p[y_-1:y_+1] += grp_d_log_f[y_] * np.array([-1, 1])
    d_log_p[-1] -= grp_d_log_f[max(y), 0]
    d2_log_p = np.zeros(2 * (len(phi),))
    d2_log_p[0, 0] += grp_d2_log_f[0, -1, -1]
    for y_ in range(1, max(y)):
        d2_log_p[y_-1:y_+1, y_-1:y_+1] += grp_d2_log_f[y_]
    d2_log_p[-1, -1] += grp_d2_log_f[max(y), 0, 0]

    return log_p, d_log_p, d2_log_p


def groupby(arr: np.ndarray, ord: np.ndarray, brk: np.ndarray, f: Callable[[np.ndarray], float]) -> np.ndarray:

    return np.array([f(a) for a in np.split(arr[ord], brk)])
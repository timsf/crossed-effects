from typing import Iterator, List, Tuple

import numpy as np

import xfx.generic.mv_conjugate
from xfx.mvglm.gibbs import eval_kernel
from xfx.generic.mv_1o_met import LatentGaussSampler
#from xfx.misc.linalg import sherman_morrison_update

from scipy.special import logsumexp


def sample_posterior(y: np.ndarray, j: np.ndarray, i: np.ndarray, 
                     tau0: np.ndarray = None, prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:

    if tau0 is None:
        tau0 = np.identity(y.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(y.shape[1], len(j))
    if prior_est_tau is None:
        prior_est_tau = len(j) * [np.identity(y.shape[1])]

    n = np.sum(y, 1)
    i_ord = np.argsort(i, 0)
    samplers = [LatentGaussSampler(n) for n in [np.bincount(i_) for i_ in i.T]]

    alp0 = np.zeros(y.shape[1])
    alp = [np.zeros((j_, y.shape[1])) for j_ in j]
    tau = prior_est_tau

    while True:
        alp0, alp = update_coefs(y, n, i, i_ord, alp0, alp, tau0, tau, samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.mv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        yield [alp0[np.newaxis]] + alp, tau


def update_coefs(y: np.ndarray, n: np.ndarray, i: np.ndarray, i_ord: np.ndarray,
                 alp0: np.ndarray, alp: List[np.ndarray], tau0: np.ndarray, tau: List[np.ndarray],
                 samplers: List[LatentGaussSampler], ome: np.random.Generator) -> Tuple[float, List[np.ndarray]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (ik_ord_, tau_, sampler_) in enumerate(zip(i_ord.T, tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y, n, i, (k_, ik_ord_), new_alp0, new_alp, tau0, tau_, sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(y: np.ndarray, n: np.ndarray, i: np.ndarray, block: Tuple[int, np.ndarray],
                       alp0: np.ndarray, alp: List[np.ndarray], tau0: np.ndarray, tau_: np.ndarray,
                       sampler: LatentGaussSampler, ome: np.random.Generator) -> Tuple[float, np.ndarray]:

    def eval_log_f(tb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.hstack([tb - talp0, np.zeros([tb.shape[0], 1])])
        a0 = np.append(talp0, 0)
        log_p, dk_log_p = eval_kernel(y, n, i, a0, alp[:k_] + [a] + alp[(k_ + 1):], eval_part, block)
        return log_p, dk_log_p[:, :-1]

    k_, _ = block
    talp_, sig_l_, ttau_, l_ttau_, u_ = remove_resid(alp[k_], tau_)
    talp0, sig_l0, ttau0, l_ttau0, _ = remove_resid(alp0[np.newaxis], tau0)

    tbet_ = talp_ + talp0
    new_tbet_ = sampler.sample(tbet_, talp0, u_, l_ttau_, eval_log_f, ome)

    mean_new_talp0 = np.linalg.solve(ttau0 + alp[k_].shape[0] * ttau_, ttau_ @ np.sum(new_tbet_, 0))
    sqrt_prec_new_talp0 = np.linalg.cholesky(ttau0 + alp[k_].shape[0] * ttau_)
    new_talp0 = mean_new_talp0 + np.linalg.solve(sqrt_prec_new_talp0, np.random.standard_normal(alp[k_].shape[1] - 1))
    new_talp_ = new_tbet_ - new_talp0

    new_alp_ = restore_resid(new_talp_, sig_l_, ttau_, ome)
    new_alp0, = restore_resid(new_talp0[np.newaxis], sig_l0, ttau0, ome)

    return new_alp0, new_alp_


def remove_resid(alp_: np.ndarray, tau_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    sig_l_ = np.linalg.solve(tau_, np.eye(tau_.shape[0])[-1])
    stack = np.vstack([-np.ones(len(sig_l_)), sig_l_, np.repeat(np.sqrt(sig_l_[-1]), len(sig_l_))])
    update = np.sum([np.outer(a_, b_) for a_, b_ in zip(stack, stack[[1, 0, 2]])], 0)
    ttau_ = np.linalg.inv((np.linalg.inv(tau_) + update)[:-1, :-1])
    l_ttau_, u_ = np.linalg.eigh(ttau_)
    talp_ = alp_[:, :-1] - alp_[:, -1][:, np.newaxis]
    return talp_, sig_l_, ttau_, l_ttau_, u_


def restore_resid(talp_: np.ndarray, sig_l_: np.ndarray, ttau_: np.ndarray, ome: np.random.Generator) -> np.ndarray:

    alp_fac = (sig_l_[:-1] - sig_l_[-1]) @ ttau_
    ralp_ = ome.normal(talp_ @ alp_fac.T, np.sqrt(sig_l_[-1] - alp_fac @ (sig_l_[:-1] - sig_l_[-1])))
    return np.hstack([ralp_[:, np.newaxis] + talp_, ralp_[:, np.newaxis]])


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    log_g = logsumexp(eta, 1)
    d_log_g = np.exp(eta - log_g[:, np.newaxis])
    return log_g, d_log_g

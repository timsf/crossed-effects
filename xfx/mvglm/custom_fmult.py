from typing import Iterator

import numpy as np
import numpy.typing as npt

import xfx.generic.mv_conjugate
from xfx.mvglm.gibbs import eval_kernel, ParamSpace
from xfx.mvglm.fmult import eval_part
from xfx.generic.mv_1o_met import LatentGaussSampler
from xfx.misc.linalg import sherman_morrison_update


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: FloatArr | None = None,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: list[FloatArr] | None = None,
    init: ParamSpace | None = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[ParamSpace]:

    if tau0 is None:
        tau0 = np.identity(y.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(y.shape[1], len(j))
    if prior_est_tau is None:
        prior_est_tau = len(j) * [np.identity(y.shape[1], dtype=np.floating)]

    if init is None:
        alp0 = np.zeros(y.shape[1])
        alp: list[FloatArr] = [np.zeros((j_, y.shape[1])) for j_ in j]
        tau = prior_est_tau
    else:
        alp, tau = init
        alp0, alp = alp[0][0], alp[1:]

    n = np.sum(y, 1)
    i_ord = np.argsort(i, 0)
    samplers = [LatentGaussSampler(j_) for j_ in j]

    while True:
        alp0, alp = update_coefs(y, n, j, i, i_ord, alp0, alp, np.zeros_like(alp0), tau0, tau, samplers, ome)
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
    alp: list[FloatArr],
    eps: FloatArr,
    tau0: FloatArr,
    tau: list[FloatArr],
    samplers: list[LatentGaussSampler],
    ome: np.random.Generator,
) -> tuple[FloatArr, list[FloatArr]]:

    new_alp0, new_alp = alp0, alp.copy()
    for k_, (tau_, sampler_) in enumerate(zip(tau, samplers)):
        new_alp0, new_alp[k_] = update_single_coef(y, n, j, i, i_ord, k_, new_alp0, new_alp, eps, tau0, tau_, sampler_, ome)
    return new_alp0, new_alp


def update_single_coef(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    i_ord: IntArr,
    k_: int,
    alp0: FloatArr,
    alp: list[FloatArr],
    eps: FloatArr,
    tau0: FloatArr,
    tau_: FloatArr,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr]:

    def eval_log_f(tb: FloatArr) -> tuple[FloatArr, FloatArr]:
        a = np.hstack([tb - talp0, np.zeros([tb.shape[0], 1])])[:, revs]
        a0 = np.append(talp0, 0)[revs]
        log_p, dk_log_p = eval_kernel(y, n, j, i, i_ord, a0 + eps, alp[:k_] + [a] + alp[(k_ + 1):], eval_part, k_)
        return log_p, dk_log_p[:, forw[:-1]]
    
    forw = ome.permutation(y.shape[1])
    revs = np.argsort(forw)

    talp_, sig_l_, ttau_ = project_coef(alp[k_][:, forw], tau_[forw][:, forw])
    talp0, sig_l0, ttau0 = project_coef(alp0[np.newaxis, forw], tau0[forw][:, forw])

    tbet_ = talp_ + talp0
    l_ttau_, u_ = np.linalg.eigh(ttau_)
    new_tbet_ = sampler.sample(tbet_, talp0, u_, l_ttau_, eval_log_f, ome)

    prec_talp0 = ttau0 + alp[k_].shape[0] * ttau_
    mean_talp0 = np.linalg.solve(prec_talp0, ttau_ @ np.sum(new_tbet_, 0))
    new_talp0 = mean_talp0 + np.linalg.solve(np.linalg.cholesky(prec_talp0), ome.standard_normal(alp[k_].shape[1] - 1))
    new_talp_ = new_tbet_ - new_talp0

    new_alp_ = restore_coef(new_talp_, sig_l_, ttau_, ome)[:, revs]
    new_alp0 = restore_coef(new_talp0[np.newaxis], sig_l0, ttau0, ome)[0, revs]

    return new_alp0, new_alp_


def project_coef(alp_: FloatArr, tau_: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    sig_11_inv = tau_[:-1, :-1] - np.outer(tau_[:-1, -1], tau_[:-1, -1]) / tau_[-1, -1]
    sig_12 = np.linalg.solve(sig_11_inv, -tau_[-1, :-1] / tau_[-1, -1])
    sig_2 = np.append(sig_12, (1 - tau_[:-1, -1] @ sig_12) / tau_[-1, -1])
    stack = np.vstack([-np.ones(len(sig_12)), sig_12, np.repeat(np.sqrt(sig_2[-1]), len(sig_12))])
    ttau_ = sherman_morrison_update(sig_11_inv, stack, stack[[1, 0, 2]])
    talp_ = alp_[:, :-1] - alp_[:, -1][:, np.newaxis]
    return talp_, sig_2, ttau_


def restore_coef(
    talp_: FloatArr,
    sig_l_: FloatArr,
    ttau_: FloatArr,
    ome: np.random.Generator,
) -> FloatArr:

    alp_fac = (sig_l_[:-1] - sig_l_[-1]) @ ttau_
    ralp_ = ome.normal(talp_ @ alp_fac.T, np.sqrt(sig_l_[-1] - alp_fac @ (sig_l_[:-1] - sig_l_[-1])))
    return np.hstack([ralp_[:, np.newaxis] + talp_, ralp_[:, np.newaxis]])

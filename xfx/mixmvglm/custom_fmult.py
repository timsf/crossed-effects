from typing import Iterator

import numpy as np
import numpy.typing as npt

import xfx.generic.mv_conjugate
import xfx.mixmvglm.fmult
import xfx.mixmvglm.gibbs
import xfx.mvglm.custom_fmult
import xfx.mvglm.gibbs
from xfx.generic.mv_2o_met import LatentGaussSampler

from scipy.linalg import block_diag


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: list[FloatArr] = None,
    init: xfx.mixmvglm.gibbs.ParamSpace = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[xfx.mixmvglm.gibbs.ParamSpace]:

    if tau0 is None:
        tau0 = np.identity(y.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(y.shape[1], len(j))
    if prior_est_tau is None:
        prior_est_tau = len(j) * [np.identity(y.shape[1])]

    if init is None:
        alp0 = np.zeros(y.shape[1])
        alp = [np.zeros((j_, y.shape[1])) for j_ in j]
        bet = np.zeros([x.shape[1], y.shape[1]])
        tau = prior_est_tau
    else:
        alp, bet, tau = init
        alp0, alp = alp[0][0], alp[1:]

    n = np.sum(y, 1)
    i_ord = np.argsort(i, 0)
    alp_samplers = [xfx.mvglm.gibbs.LatentGaussSampler(j_) for j_ in j]
    bet_sampler = LatentGaussSampler(1)

    while True:
        alp0, alp = xfx.mvglm.custom_fmult.update_coefs(y, n, j, i, i_ord, alp0, alp, xfx.mixmvglm.gibbs.eval_reg_pred(x, np.zeros_like(alp0), bet), tau0, tau, alp_samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.mv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        alp0, bet = update_coefs(y, n, x, alp0, bet, xfx.mvglm.gibbs.eval_lin_pred(i, np.zeros_like(alp0), alp), tau0, bet_sampler, ome)
        yield [alp0[np.newaxis]] + alp, bet, tau


def update_coefs(
    y: FloatArr,
    n: FloatArr,
    x: FloatArr,
    alp0: FloatArr,
    bet: FloatArr,
    eps: FloatArr,
    tau0: FloatArr,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr]:

    def eval_log_f(tb: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
        b = np.append(np.reshape(tb, (tgam.shape)), np.zeros((tgam.shape[0], 1)), 1)
        log_p, d_log_p, d2_log_p = xfx.mixmvglm.gibbs.eval_kernel(y, n, z, eps, b, xfx.mixmvglm.fmult.eval_part2)
        return (
            log_p[np.newaxis], 
            np.hstack(d_log_p[:,:-1])[np.newaxis], 
            np.vstack(np.dstack(d2_log_p[:,:,:-1,:-1])),
        )

    gam = np.append(alp0[np.newaxis], bet, 0)
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    tgam, sig_l_, ttau_ = xfx.mvglm.custom_fmult.project_coef(gam, tau0)
    # l_ttau_, u_ = np.linalg.eigh(ttau_)
    # l_ttau = np.tile(l_ttau_, gam.shape[0])
    # u = block_diag(*(gam.shape[0] * [u_]))
    ttau = block_diag(*(gam.shape[0] * [ttau_]))
    new_tgam = np.reshape(sampler.sample(tgam.flatten()[np.newaxis], np.zeros(np.prod(tgam.shape)), ttau, eval_log_f, ome), tgam.shape)
    new_gam = xfx.mvglm.custom_fmult.restore_coef(new_tgam, sig_l_, ttau_, ome)
    new_alp0, new_bet = new_gam[0], new_gam[1:]
    return new_alp0, new_bet

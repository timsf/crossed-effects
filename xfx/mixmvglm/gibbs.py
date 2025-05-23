from typing import Iterator

import numpy as np
import numpy.typing as npt

import xfx.mixmvglm.generic
import xfx.mvglm.generic
import xfx.mvglm.gibbs
import xfx.generic.mv_conjugate
from xfx.generic.mv_2o_met import LatentGaussSampler

from scipy.linalg import block_diag


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]
ParamSpace = tuple[list[FloatArr], FloatArr, list[FloatArr]]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    eval_part: xfx.mvglm.generic.PartFunc,
    eval_part2: xfx.mixmvglm.generic.PartFunc2,
    tau0: FloatArr | None,
    prior_n_tau: FloatArr | None,
    prior_est_tau: list[FloatArr] | None,
    init: ParamSpace | None,
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
        bet = np.zeros((x.shape[1], y.shape[1]))
        tau = prior_est_tau
    else:
        alp, bet, tau = init
        alp0, alp = alp[0][0], alp[1:]

    i_ord = np.argsort(i, 0)
    alp0_sampler = xfx.mvglm.gibbs.LatentGaussSampler(1)
    alp_samplers = [xfx.mvglm.gibbs.LatentGaussSampler(j_) for j_ in j]
    bet_sampler = LatentGaussSampler(1)

    while True:
        alp0, alp = xfx.mvglm.gibbs.update_coefs(y, n, j, i, i_ord, alp0, alp, eval_reg_pred(x, np.zeros_like(alp0), bet), tau0, tau, eval_part, alp0_sampler, alp_samplers, True, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.mv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        alp0, bet = update_coefs(y, n, x, alp0, bet, xfx.mvglm.gibbs.eval_lin_pred(i, np.zeros_like(alp0), alp), tau0, eval_part2, bet_sampler, ome)
        yield [alp0[np.newaxis]] + alp, bet, tau


def update_coefs(
    y: FloatArr,
    n: FloatArr,
    x: FloatArr,
    alp0: FloatArr,
    bet: FloatArr,
    eps: FloatArr,
    tau0: FloatArr,
    eval_part2: xfx.mixmvglm.generic.PartFunc2,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> tuple[FloatArr, FloatArr]:

    def eval_log_f(b: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
        g = np.reshape(b, (gam.shape))
        log_p, d_log_p, d2_log_p = eval_kernel(y, n, z, eps, g, eval_part2)
        return (
            np.array(log_p)[np.newaxis], 
            np.hstack(d_log_p)[np.newaxis], 
            np.vstack(np.dstack(d2_log_p)),
        )

    gam = np.append(alp0[np.newaxis], bet, 0)
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    tau = block_diag(*(gam.shape[0] * [tau0]))
    new_gam = np.reshape(sampler.sample(gam.flatten()[np.newaxis], np.zeros(np.prod(gam.shape)), tau, eval_log_f, ome), gam.shape)
    new_alp0, new_bet = new_gam[0], new_gam[1:]
    return new_alp0, new_bet


def eval_kernel(
    y: FloatArr,
    n: FloatArr,
    x: FloatArr,
    alp0: FloatArr,
    bet: FloatArr,
    eval_part2: xfx.mixmvglm.generic.PartFunc2,
) -> tuple[float, FloatArr, FloatArr]:

    eta = eval_reg_pred(x, alp0, bet)
    log_f, d_log_f, d2_log_f = xfx.mixmvglm.generic.eval_densities2(y, n, eta, eval_part2)
    log_p = sum(log_f)
    d_log_p = x.T @ d_log_f
    d2_log_p = np.einsum('ij,ik,ilm->jklm', x, x, d2_log_f, optimize=True)
    return log_p, d_log_p, d2_log_p


def eval_reg_pred(x: FloatArr, alp0: FloatArr, bet: FloatArr) -> FloatArr:

    return alp0 + x @ bet

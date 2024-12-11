from typing import Iterator

import numpy as np
import numpy.typing as npt

import xfx.glm.generic
import xfx.glm.gibbs
import xfx.generic.uv_conjugate
from xfx.generic.mv_2o_met import LatentGaussSampler


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
ParamSpace = tuple[list[FloatArr], FloatArr, FloatArr]
DispParamSpace = tuple[list[FloatArr], FloatArr, FloatArr, float]


def sample_reglr_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    eval_part: xfx.glm.generic.PartFunc,
    tau0: float,
    prior_n_tau: FloatArr | None,
    prior_est_tau: FloatArr | None,
    init: ParamSpace | None,
    ome: np.random.Generator,
) -> Iterator[ParamSpace]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))

    if init is None:
        alp0 = 0
        alp = [np.zeros(j_) for j_ in j]
        bet = np.zeros(x.shape[1])
        tau = prior_est_tau
    else:
        alp, bet, tau = init
        alp0, alp = alp[0][0], alp[1:]

    i_ord = np.argsort(i, 0)
    alp_samplers = [xfx.glm.gibbs.LatentGaussSampler(j_) for j_ in j]
    bet_sampler = LatentGaussSampler(1)

    while True:
        alp0, alp = xfx.glm.gibbs.update_coefs(y, n, j, i, i_ord, alp0, alp, eval_reg_pred(x, 0, bet), tau0, tau, 1, True, eval_part, alp_samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        alp0, bet = update_coefs(y, n, x, alp0, bet, xfx.glm.gibbs.eval_lin_pred(i, 0, alp), tau0, 1, eval_part, bet_sampler, ome)
        yield [np.array([alp0])] + alp, bet, tau


def sample_disp_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    eval_part: xfx.glm.generic.PartFunc,
    eval_base: xfx.glm.generic.BaseFunc,
    tau0: float,
    prior_n_tau: FloatArr | None,
    prior_est_tau: FloatArr | None,
    prior_n_phi: float | None,
    prior_est_phi: float | None,
    init: DispParamSpace | None,
    ome: np.random.Generator,
) -> Iterator[DispParamSpace]:

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
        bet = np.zeros(x.shape[1])
        tau = prior_est_tau
        phi = 1
    else:
        alp, bet, tau, phi = init
        alp0, alp = alp[0][0], alp[1:]

    i_ord = np.argsort(i, 0)
    alp_samplers = [xfx.glm.gibbs.LatentGaussSampler(j_) for j_ in j]
    bet_sampler = LatentGaussSampler(1)

    while True:
        alp0, alp = xfx.glm.gibbs.update_coefs(y1, n, j, i, i_ord, alp0, alp, eval_reg_pred(x, 0, bet), tau0, tau, phi, True, eval_part, alp_samplers, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        alp0, bet = update_coefs(y1, n, x, alp0, bet, xfx.glm.gibbs.eval_lin_pred(i, 0, alp), tau0, phi, eval_part, bet_sampler, ome)
        phi = xfx.glm.generic.update_dispersion(y1, y2, n, eval_reg_pred(x, 0, bet) + xfx.glm.gibbs.eval_lin_pred(i, alp0, alp), phi, eval_part, eval_base, prior_n_phi, prior_est_phi, ome)
        yield [np.array([alp0])] + alp, bet, tau, phi


def update_coefs(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    alp0: float,
    bet: FloatArr,
    eps: float,
    tau0: float,
    phi: float,
    eval_part: xfx.glm.generic.PartFunc,
    sampler: LatentGaussSampler,
    ome: np.random.Generator,
) -> tuple[float, FloatArr]:

    def eval_log_p(b: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:
        log_p, dk_log_p, d2k_log_p = eval_kernel(y1, n, z, eps, b[0], eval_part)
        return log_p[np.newaxis] / phi, dk_log_p[np.newaxis] / phi, d2k_log_p / phi

    gam = np.append(alp0, bet)
    prec = np.append(tau0, np.zeros_like(bet))
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    new_gam = sampler.sample(gam[np.newaxis], np.zeros_like(gam), prec, eval_log_p, ome)[0]
    new_alp0, new_bet = new_gam[0], new_gam[1:]
    return new_alp0, new_bet


def eval_kernel(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    alp0: float,
    bet: FloatArr,
    eval_part: xfx.glm.generic.PartFunc,
) -> tuple[float, FloatArr, FloatArr]:

    eta = eval_reg_pred(x, alp0, bet)
    log_f, d_log_f, d2_log_f = xfx.glm.generic.eval_densities(y1, n, eta, eval_part)
    log_p = np.sum(log_f)
    d_log_p = x.T @ d_log_f
    d2_log_p = np.einsum('ij,ik,i->jk', x, x, d2_log_f, optimize=True)
    return log_p, d_log_p, d2_log_p


def eval_reg_pred(x: FloatArr, alp0: float, bet: FloatArr) -> FloatArr:

    return alp0 + x @ bet

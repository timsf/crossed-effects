from typing import Iterator

import numpy as np
import numpy.typing as npt

import xfx.lm.gibbs
import xfx.glm.custom_pg_binomial
import xfx.generic.uv_conjugate
import xfx.generic.reg_conjugate


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]
ParamSpace = tuple[list[FloatArr], FloatArr, FloatArr, FloatArr]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr | None = None,
    tau0: float = 0,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: FloatArr | None = None,
    init: ParamSpace | None = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[ParamSpace]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))

    if x is None:
        x = np.zeros([y.shape[0], 0])

    if init is None:
        alp0 = 0
        alp: list[FloatArr] = [np.zeros(j_) for j_ in j]
        bet = np.zeros(x.shape[1])
        tau = prior_est_tau
        nu = np.ones_like(y)
    else:
        alp, bet, tau, nu = init
        alp0, alp = alp[0][0], alp[1:]

    while True:
        _, yr1, yr2 = xfx.lm.gibbs.reduce_data(y - nu * (x @ bet) - n / 2, np.zeros_like(y), nu, i)
        alp = xfx.lm.gibbs.update_coefs(yr1, yr2, None, alp, tau0, tau, 1, ome)
        # alp0 = xfx.lm.gibbs.update_intercept(yr0, yr1, alp, 1, ome)
        alp0, bet = update_coefs(y, x, n, np.sum([alp_[i_] for alp_, i_ in zip(alp, i.T)], 0), nu, tau0, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        nu = xfx.glm.custom_pg_binomial.update_latent(n, alp0 + sum([alp_[i_] for alp_, i_ in zip(alp, i.T)]) + x @ bet, True, ome)
        yield [np.array([alp0])] + alp, bet, tau, nu


def update_coefs(
    y: FloatArr,
    x: FloatArr,
    n: FloatArr,
    eps: FloatArr,
    nu: FloatArr,
    tau0: float,
    ome: np.random.Generator,
) -> tuple[float, FloatArr]:

    z = np.hstack([np.ones([x.shape[0], 1]), x])
    prec = np.append(tau0, np.zeros(x.shape[1]))
    bet_mean, bet_prec = xfx.generic.reg_conjugate.est_gls((y - n / 2) / nu - eps, z, np.diag(nu), np.diag(prec))
    bet = bet_mean + np.linalg.solve(np.linalg.cholesky(bet_prec).T, ome.standard_normal(x.shape[1] + 1))
    return bet[0], bet[1:]

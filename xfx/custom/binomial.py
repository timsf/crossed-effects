from typing import Iterator, List, Tuple
from itertools import count

import numpy as np
import numpy.typing as npt
from scipy.stats import invgauss

import xfx.lm.gibbs
import xfx.generic.uv_conjugate


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(y: FloatArr, n: FloatArr, j: IntArr, i: IntArr,
                     prior_n_tau: FloatArr = None, prior_est_tau: FloatArr = None,
                     init: Tuple[List[FloatArr], FloatArr, FloatArr] = None,
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[FloatArr], FloatArr, FloatArr]]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))

    if init is None:
        alp0 = 0
        alp = [np.zeros(j_) for j_ in j]
        tau = prior_est_tau
        nu = np.ones_like(y)
    else:
        alp, tau, nu = init
        alp0, alp = alp[0][0], alp[1:]

    while True:
        x0, x1, x2 = xfx.lm.gibbs.reduce_data(y - n / 2, np.zeros_like(y), nu, i)
        alp = xfx.lm.gibbs.update_coefs(x1, x2, None, alp, tau, 1, ome)
        alp0 = xfx.lm.gibbs.update_intercept(x0, x1, alp, 1, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        nu = update_latent(n, i, alp0, alp, ome)
        yield [np.array([alp0])] + alp, tau, nu


def update_latent(n: FloatArr, i: IntArr, alp0: float, alp: List[FloatArr], ome: np.random.Generator) -> FloatArr:

    eta = np.float_(alp0 + sum([alp_[i_] for alp_, i_ in zip(alp, i.T)]))
    return np.array([sample_pg(n_, eta_, ome) for n_, eta_ in zip(n, eta)])


def sample_pg(a: int, b: float, ome: np.random.Generator) -> float:

    return sum([sample_spg(b, ome) for _ in range(a)])


def sample_spg(b: float, ome: np.random.Generator) -> float:

    return sample_jacobi(b / 2, ome) / 4


def sample_jacobi(z: float, ome: np.random.Generator, t: float = .64) -> float:

    th1 = 1 / np.abs(z)
    th2 = z**2/2 + np.pi**2/8
    log_p1 = np.log(2) - np.abs(z) + invgauss(th1).logcdf(t)
    log_p2 = np.log(np.pi) - th2 * t - np.log(2 * th2)
    log_p = log_p1 - np.logaddexp(log_p1, log_p2)
    while True:
        if np.log(ome.uniform()) < log_p:
            x = sample_rtrunc_invgauss(th1, t, ome)
        else:
            x = sample_ltrunc_exp(th2, t, ome)
        series = series_jacobi(x, z, t)
        u = ome.uniform(0, next(series))
        for n, s in enumerate(series, 1):
            if bool(n % 2) and u <= s:
                return x
            if not bool(n % 2) and s < u:
                break


def sample_rtrunc_invgauss(mu: float, t: float, ome: np.random.Generator) -> float:

    if mu > t:
        while True:
            while True:
                e1, e2 = ome.exponential(size=2)
                if e1 ** 2 <= 2 * e2 / t:
                    break
            x = t / (1 + t * e1) ** 2
            if np.log(ome.uniform()) <= -x / (2 * mu ** 2):
                return x
    else:
        while True:
            x = ome.wald(mu, 1)
            if x < t:
                return x


def sample_ltrunc_exp(lam: float, t: float, ome: np.random.Generator) -> float:

    return t + ome.exponential() / lam


def series_jacobi(x: float, z: float, t: float) -> Iterator[float]:

    s = 0
    tilt = np.cosh(z) * np.exp(-z**2 * x / 2)
    for n in count(0):
        if x < t:
            s += (-1) ** n * np.pi * (n + 1/2) * np.exp(-(2 / x) * (n + 1/2) ** 2) * (2 / (np.pi * x)) ** (3 / 2)
        else:
            s += (-1) ** n * np.pi * (n + 1/2) * np.exp(-(x / 2) * ((n + 1/2) * np.pi) ** 2)
        yield tilt * s

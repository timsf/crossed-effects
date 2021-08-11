from typing import Iterator, List, Tuple
from itertools import count

import numpy as np
from scipy.stats import invgauss

import xfx.lm.gibbs
import xfx.generic.uv_conjugate


def sample_posterior(y: np.ndarray, n: np.ndarray, j: np.ndarray, i: np.ndarray,
                     prior_n_tau: np.ndarray = None, prior_est_tau: np.ndarray = None,
                     init: Tuple[List[np.ndarray], np.ndarray, np.ndarray] = None,
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], np.ndarray, float]]:

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
        alp0, alp, tau, nu = init

    while True:
        x0, x1, x2 = xfx.lm.gibbs.reduce_data((y - n / 2) / nu, np.zeros_like(y), nu, i)
        alp = xfx.lm.gibbs.update_coefs(x1, x2, None, alp, tau, 1, ome)
        alp0 = xfx.lm.gibbs.update_intercept(x0, x1, alp, 1, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        nu = update_latent(n, i, alp0, alp, ome)
        yield [np.array([alp0])] + alp, tau, nu


def update_latent(n: np.ndarray, i: np.ndarray, alp0: float, alp: List[np.ndarray], ome: np.random.Generator) -> np.ndarray:

    eta = alp0 + sum([alp_[i_] for alp_, i_ in zip(alp, i.T)])
    return np.array([sample_pg(n_, eta_, ome) for n_, eta_ in zip(n, eta)])


def sample_pg(a: int, b: float, ome: np.random.Generator) -> float:

    return sum([sample_spg(b, ome) for _ in range(a)])


def sample_spg(b: float, ome: np.random.Generator) -> float:

    return sample_jacobi(b / 2, ome) / 4


def sample_jacobi(z: float, ome: np.random.Generator, t: float = .64) -> float:

    th1 = 1 / np.abs(z)
    th2 = z**2/2 + np.pi**2/8
    p1 = 2 * np.exp(-np.abs(z)) * invgauss(th1).cdf(t)
    p2 = np.pi * np.exp(-th2 * t) / (2 * th2)
    p = p1 / (p1 + p2)
    while True:
        if ome.uniform() < p:
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
            if ome.uniform() <= np.exp(-x / (2 * mu ** 2)):
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

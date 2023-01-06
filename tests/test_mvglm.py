import pytest

import numpy as np
from scipy.special import softmax
from scipy.stats import wishart

import xfx.mvglm.cmult
import xfx.custom.symmetric_multinomial


def sample_coef_fixture(j, tau, ome):

    alp = [ome.multivariate_normal(np.zeros(tau_.shape[0]), np.linalg.inv(tau_), j_) for tau_, j_ in zip(tau, j)]
    return [np.hstack([alp_, np.zeros([alp_.shape[0], 1])]) for alp_ in alp]


def sample_randfx_fixture(l, j, df_tau, scale_tau, ome):

    tau = [wishart.rvs(df_tau, scale_tau * np.identity(l - 1), random_state=ome) for _ in range(len(j))]
    alp = sample_coef_fixture(j, tau, ome)
    return alp, tau


def sample_data_fixture(i, n_inflator, alp0, alp, ome):

    return (alp0 + sum([alp_[j_] for alp_, j_ in zip(alp, i.T)])) * n_inflator, np.repeat(n_inflator, i.shape[0])


def sample_balanced_design(j, ome):

    n = 1
    for j_ in j:
        n = np.lcm(n, j_)
    i = np.array([np.repeat(np.arange(j_), n / j_) for j_ in j]).T
    for k_ in range(len(j)):
        ome.shuffle(i[:, k_])
    return i


def sample_balanced_fixture(l, j, df_tau, scale_tau, n_inflator, ome):

    alp0 = np.zeros(l)
    alp, tau = sample_randfx_fixture(l, j, df_tau, scale_tau, ome)
    i = sample_balanced_design(j, ome)
    y, n = sample_data_fixture(i, n_inflator, alp0, alp, ome)
    return (y, n, i), (alp0, alp, tau)


def test_multinomial(j=np.array([2, 3]), l=3, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng()
    data, params = sample_balanced_fixture(l, j, 2 * l, 1, n_inflator, ome)
    y, n, i = data
    y1 = np.array([ome.multinomial(n_, p_) for n_, p_ in zip(n, softmax(y / n[:, np.newaxis], 1))])
    sampler = xfx.mvglm.cmult.sample_posterior(y1, j, i, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_symmetric_multinomial(j=np.array([2, 3]), l=3, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng()
    data, params = sample_balanced_fixture(l, j, 2 * l, 1, n_inflator, ome)
    y, n, i = data
    y1 = np.array([ome.multinomial(n_, p_) for n_, p_ in zip(n, softmax(y / n[:, np.newaxis], 1))])
    sampler = xfx.custom.symmetric_multinomial.sample_posterior(y1, j, i, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]

import pytest

import numpy as np
from scipy.special import expit

import tests.test_glm
import xfx.mixglm.binomial
import xfx.mixglm.gaussian
import xfx.mixglm.poisson
import xfx.mixglm.custom_pg_binomial


def sample_regression_fixture(n, m, ome=np.random.default_rng()):
    x = ome.standard_normal((n, m))
    bet = ome.standard_normal(m)
    return x, bet


def sample_balanced_fixture(j, m, alp0=0, df_tau=2, scale_tau=1, ome=np.random.default_rng()):

    alp, tau = tests.test_glm.sample_randfx_fixture(j, df_tau, scale_tau, ome)
    i = tests.test_glm.sample_balanced_design(j, ome)
    x, bet = sample_regression_fixture(np.prod(j), m)
    eta = alp0 + np.sum([alp_[j_] for alp_, j_ in zip(alp, i.T)], 0) + x @ bet
    return (eta, i, x), (alp0, alp, bet, tau)


def test_gaussian(j=np.array([2, 3]), m=2, lam=1, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, m, ome=ome)
    eta, i, x = data
    y = ome.normal(eta[:, np.newaxis], 1 / np.sqrt(lam), size=(i.shape[0], n_inflator))
    n = np.repeat(n_inflator, len(eta))
    y1 = np.sum(y, 1)
    y2 = np.sum(np.square(y), 1)
    sampler = xfx.mixglm.gaussian.sample_posterior(y1, y2, n, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_binomial(j=np.array([2, 3]), m=2, n_inflator=int(1e3), n_samples=int(1e4), seed=4):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, m, ome=ome)
    eta, i, x = data
    n = np.repeat(n_inflator, len(eta))
    y1 = ome.binomial(n, expit(eta))
    sampler = xfx.mixglm.binomial.sample_posterior(y1, n, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_poisson(j=np.array([2, 3]), m=2, n_inflator=int(1e3), n_samples=int(1e4), seed=4):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, m, ome=ome)
    eta, i, x = data
    n = np.repeat(n_inflator, len(eta))
    y1 = n * ome.poisson(np.exp(eta))
    sampler = xfx.mixglm.poisson.sample_posterior(y1, n, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_custom_pg_binomial(j=np.array([2, 3]), n_inflator=int(1e3), n_samples=int(1e4), seed=4):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, 0, ome=ome)
    eta, i, x = data
    n = np.repeat(n_inflator, len(eta))
    y1 = ome.binomial(n, expit(eta))
    sampler = xfx.mixglm.custom_pg_binomial.sample_posterior(y1, n, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


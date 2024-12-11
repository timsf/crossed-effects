import pytest

import numpy as np
from scipy.special import softmax

import tests.test_mvglm
import xfx.mixmvglm.fmult
import xfx.mixmvglm.custom_fmult


def sample_regression_fixture(n, m, l, ome=np.random.default_rng()):
    x = ome.standard_normal((n, m))
    bet = ome.standard_normal((m, l))
    return x, bet


def sample_balanced_fixture(l, j, m, df_tau, scale_tau, n_inflator, ome):

    alp0 = np.zeros(l)
    alp, tau = tests.test_mvglm.sample_randfx_fixture(l, j, df_tau, scale_tau, ome)
    i = tests.test_mvglm.sample_balanced_design(j, ome)
    eta, n = tests.test_mvglm.sample_data_fixture(i, n_inflator, alp0, alp, ome)
    x, bet = sample_regression_fixture(np.prod(j), m, l)
    return (eta + x @ bet, n, i, x), (alp0, alp, bet, tau)


def test_fmult(l=3, j=np.array([2, 3]), m=2, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(l, j, m, 2 * l, 1, n_inflator, ome)
    eta, n, i, x = data
    y1 = np.array([ome.multinomial(n_, p_) for n_, p_ in zip(n, softmax(eta / n[:, np.newaxis], 1))])
    sampler = xfx.mixmvglm.fmult.sample_posterior(y1, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_custom_fmult(l=3, j=np.array([2, 3]), m=2, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(l, j, m, 2 * l, 1, n_inflator, ome)
    eta, n, i, x = data
    y1 = np.array([ome.multinomial(n_, p_) for n_, p_ in zip(n, softmax(eta / n[:, np.newaxis], 1))])
    sampler = xfx.mixmvglm.custom_fmult.sample_posterior(y1, j, i, x, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]

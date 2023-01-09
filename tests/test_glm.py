import pytest

import numpy as np
from scipy.special import expit

import xfx.glm.binomial
import xfx.glm.gaussian
import xfx.custom.binomial


def sample_coef_fixture(j, tau, ome):

    alp = [ome.normal(0, 1 / np.sqrt(tau_), j_) for tau_, j_ in zip(tau, j)]
    return [alp_ - np.mean(alp_) for alp_ in alp]


def sample_randfx_fixture(i, df_tau, scale_tau, ome):

    tau = scale_tau * ome.chisquare(df_tau, len(i))
    alp = sample_coef_fixture(i, tau, ome)
    return alp, tau

def sample_balanced_design(j, ome):

    n = 1
    for j_ in j:
        n = np.lcm(n, j_)
    i = np.array([np.repeat(np.arange(j_), n / j_) for j_ in j]).T
    ome.shuffle(i, 0)
    return i


def sample_mar_design(j, p_miss, ome):

    i = np.stack(np.meshgrid(*[np.arange(j_) for j_ in j])).T.reshape(-1, 2)
    i = i[ome.uniform(size=i.shape[0]) > p_miss]
    ome.shuffle(i, 0)
    return i


def sample_balanced_fixture(j, alp0=0, df_tau=2, scale_tau=1, ome=np.random.default_rng()):

    alp, tau = sample_randfx_fixture(j, df_tau, scale_tau, ome)
    i = sample_balanced_design(j, ome)
    eta = alp0 + np.sum([alp_[j_] for alp_, j_ in zip(alp, i.T)], 0)
    return (eta, i), (alp0, alp, tau)


def sample_mar_fixture(j, df_tau=2, scale_tau=1, p_miss=.1, ome=np.random.default_rng()):

    alp0 = 0
    alp, tau = sample_randfx_fixture(j, df_tau, scale_tau, ome)
    i = sample_mar_design(j, p_miss, ome)
    eta = alp0 + np.sum([alp_[j_] for alp_, j_ in zip(alp, i.T)], 0)
    return (eta, i), (alp0, alp, tau)


def test_gaussian(j=np.array([2, 3]), lam=1, n_inflator=int(1e3), n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, ome=ome)
    eta, i = data
    y = ome.normal(eta[:, np.newaxis], 1 / np.sqrt(lam), size=(i.shape[0], n_inflator))
    n = np.repeat(n_inflator, len(eta))
    y1 = np.sum(y, 1)
    y2 = np.sum(np.square(y), 1)
    sampler = xfx.glm.gaussian.sample_posterior(y1, y2, n, j, i, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_binomial(j=np.array([2, 3]), n_inflator=int(1e3), n_samples=int(1e4), seed=4):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, 0, ome=ome)
    eta, i = data
    n = np.repeat(n_inflator, len(eta))
    y1 = ome.binomial(n, expit(eta))
    sampler = xfx.glm.binomial.sample_posterior(y1, n, j, i, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]


def test_aug_binomial(j=np.array([2, 3]), n_inflator=int(1e3), n_samples=int(1e4), seed=4):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, 0, ome=ome)
    eta, i = data
    n = np.repeat(n_inflator, len(eta))
    y1 = ome.binomial(n, expit(eta))
    sampler = xfx.custom.binomial.sample_posterior(y1, n, j, i, ome=ome, collapse=True)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]

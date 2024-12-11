import pytest

import numpy as np
from scipy.special import expit

import xfx.olm.logit


def sample_coef_fixture(j, tau, ome):

    alp = [ome.normal(0, 1 / np.sqrt(tau_), j_) for tau_, j_ in zip(tau, j)]
    return [alp_ - np.mean(alp_) for alp_ in alp]


def sample_randfx_fixture(i, df_tau, scale_tau, ome):

    tau = scale_tau * ome.chisquare(df_tau, len(i))
    alp = sample_coef_fixture(i, tau, ome)
    return alp, tau


def sample_threshold_fixture(l, ome):

    return np.cumsum(np.abs(ome.standard_normal(l - 1)))


def sample_balanced_design(j, ome):

    n = 1
    for j_ in j:
        n = np.lcm(n, j_)
    i = np.array([np.repeat(np.arange(j_), n / j_) for j_ in j]).T
    ome.shuffle(i, 0)
    return i


def sample_balanced_fixture(eval_cdf, j, l=3, alp0=0, lam=1, df_tau=2, scale_tau=1, n_inflator=int(1e3), ome=np.random.default_rng()):

    alp, tau = sample_randfx_fixture(j, df_tau, scale_tau, ome)
    i = sample_balanced_design(j, ome)
    ups = sample_threshold_fixture(l, ome)
    eta = alp0 + np.sum([alp_[j_] for alp_, j_ in zip(alp, i.T)], 0)
    cdf = np.hstack([np.zeros_like(eta[:, np.newaxis]), eval_cdf(ups - ups[0] - eta[:, np.newaxis]), np.ones_like(eta[:, np.newaxis])])
    pmf = np.diff(cdf, 1)
    y = np.array([ome.multinomial(n_inflator, pmf_) for pmf_ in pmf])
    ylong = np.hstack([np.repeat(np.arange(l), y_) for y_ in y])
    ilong = np.repeat(i, n_inflator, 0)
    return (ylong, ilong), (alp0, alp, tau, ups)


def test_logit(j=np.array([2, 3]), l=5, n_inflator=int(1e2), n_samples=int(1e3), seed=0):

    eval_cdf = expit
    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(eval_cdf, j, l, n_inflator=n_inflator, ome=ome)
    y, i = data
    sampler = xfx.olm.logit.sample_posterior(y, j, i, ome=ome)
    samples = [x_ for _, x_ in zip(range(n_samples), sampler)]

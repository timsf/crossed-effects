import pytest

import numpy as np
from scipy.stats import kstest, chi2

from xfx.lm import gibbs
from xfx.lm import iid


def sample_coef_fixture(j, tau, ome):

    alp = [ome.normal(0, 1 / np.sqrt(tau_), j_) for tau_, j_ in zip(tau, j)]
    return [alp_ - np.mean(alp_) for alp_ in alp]


def sample_randfx_fixture(i, df_tau, scale_tau, ome):

    tau = scale_tau * ome.chisquare(df_tau, len(i))
    alp = sample_coef_fixture(i, tau, ome)
    return alp, tau


def sample_data_fixture(i, n_inflator, alp0, alp, lam, ome):

    y = ome.normal((alp0 + np.sum([alp_[j_] for alp_, j_ in zip(alp, i.T)], 0))[:, np.newaxis], 
                   1 / np.sqrt(lam), size=(i.shape[0], n_inflator))
    y1 = np.sum(y, 1)
    y2 = np.sum(np.square(y), 1)
    n = np.repeat(n_inflator, i.shape[0])
    return y1, y2, n


def sample_balanced_design(j, ome):

    n = 1
    for j_ in j:
        n = np.lcm(n, j_)
    i = np.array([np.repeat(np.arange(j_), n / j_) for j_ in j]).T
    for k_ in range(len(j)):
        ome.shuffle(i[:, k_])
    return i


def sample_balanced_fixture(j, alp0=0, lam=1, df_tau=2, scale_tau=1, n_inflator=1, ome=np.random.default_rng()):

    alp, tau = sample_randfx_fixture(j, df_tau, scale_tau, ome)
    i = sample_balanced_design(j, ome)
    y1, y2, n = sample_data_fixture(i, n_inflator, alp0, alp, lam, ome)
    return (y1, y2, n, i), (alp0, alp, tau, lam)


def test_benchmark(j=np.array([2, 3]), n_inflator=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, n_inflator=n_inflator, ome=ome)
    y1, _, n, i = data
    post_mean, post_cov = iid.update_coefs(y1, n, j, i, params[2], params[3])


def test_gibbs(j=np.array([2, 3]), n_samples=int(1e3), n_inflator=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, n_inflator=n_inflator, ome=ome)
    y1, y2, n, i = data
    sampler = gibbs.sample_posterior(y1, y2, n, j, i, ome=ome)
    samples = [sample[1] for _, sample in zip(range(n_samples), sampler)]


def test_benchmark_vs_gibbs(j=np.array([2, 3]), n_inflator=int(1e6), n_samples=int(1e3), alpha=1e-2, seed=0):

    ome = np.random.default_rng(seed)
    data, params = sample_balanced_fixture(j, n_inflator=n_inflator, ome=ome)
    y1, y2, n, i = data

    sampler = gibbs.sample_posterior(y1, y2, n, j, i, np.repeat(np.inf, len(j)), params[2], np.inf, params[3], ome=ome)
    samples = np.vstack([np.hstack(sample[0][1:]) for _, sample in zip(range(n_samples), sampler)])
    post_mean, post_cov = iid.update_coefs(y1, n, j, i, params[2], params[3])
    post_prec = np.linalg.inv(post_cov)
    mahdist = np.array([(s - post_mean) @ post_prec @ (s - post_mean) for s in samples])[1:]
    assert kstest(mahdist, chi2(samples.shape[1]).cdf)[1] > alpha

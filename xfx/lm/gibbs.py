import itertools as it
from typing import Iterator, NamedTuple
from math import sqrt

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import xfx.generic.uv_conjugate


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]
ParamSpace = tuple[list[FloatArr], FloatArr, float]


class SuffStat0(NamedTuple):
    len: float
    sum: float
    sum2: float


class SuffStat1(NamedTuple):
    len: FloatArr
    sum: FloatArr


class SuffStat2(NamedTuple):
    len: csr_matrix
    sum: csr_matrix


def sample_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: float = 0,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: FloatArr | None = None,
    prior_n_lam: float = 1,
    prior_est_lam: float = 1,
    init: ParamSpace | None = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[ParamSpace]:

    if prior_n_tau is None:
        prior_n_tau = np.ones(len(j))
    if prior_est_tau is None:
        prior_est_tau = np.ones(len(j))

    if init is None:
        alp0 = 0
        alp: list[FloatArr] = [np.zeros(j_) for j_ in j]
        tau = prior_est_tau
        lam = prior_est_lam
    else:
        alp, tau, lam = init
        alp0, alp = alp[0][0], alp[1:]

    x0, x1, x2 = reduce_data(y1, y2, n, i)

    while True:
        alp = update_coefs(x1, x2, None if collapse else alp0, alp, tau0, tau, lam, ome)
        alp0 = update_intercept(x0, x1, alp, tau0, lam, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = xfx.generic.uv_conjugate.update_factor_precision(j, alp, prior_n_tau, prior_est_tau, ome)
        if not np.isinf(prior_n_lam):
            lam = update_resid_precision(x0, x1, x2, alp0, alp, prior_n_lam, prior_est_lam, ome)
        yield [np.array([alp0])] + alp, tau, lam


def update_coefs(
    x1: list[SuffStat1],
    x2: dict[tuple[int, int], SuffStat2],
    alp0: float | None,
    alp: list[FloatArr],
    tau0: float,
    tau: FloatArr,
    lam: float,
    ome: np.random.Generator,
) -> list[FloatArr]:

    alp_new = alp.copy()
    for k_, (x1_, tau_) in enumerate(zip(x1, tau)):
        x2_sub = [SuffStat2(v.len, v.sum) if k_ == k[0] else SuffStat2(v.len.T, v.sum.T)
                  for k, v in x2.items() if k_ in k]
        alp_sub = alp_new[:k_] + alp_new[(k_ + 1):]
        if alp0 is None:
            alp0_ = update_intercept_collapsed(x1_, x2_sub, alp_sub, tau0, tau_, lam, ome)
        else:
            alp0_ = alp0
        alp_new[k_] = update_coefs_single(x1_, x2_sub, alp0_, alp_sub, tau_, lam, ome)
    return alp_new


def update_intercept(
    x0: SuffStat0,
    x1: list[SuffStat1],
    alp: list[FloatArr],
    lam: float,
    tau0: float,
    ome: np.random.Generator,
) -> float:

    fitted_sum = float(sum([alp_ @ x1_.len for alp_, x1_ in zip(alp, x1)]))
    post_prec = x0.len * lam + tau0
    post_mean = (x0.sum - fitted_sum) * lam / post_prec
    post_sd = 1 / sqrt(post_prec)
    return ome.normal(post_mean, post_sd)


def update_intercept_collapsed(
    x1_: SuffStat1,
    x2_sub: list[SuffStat2],
    alp_sub: list[FloatArr],
    tau0: float,
    tau_: float,
    lam: float,
    ome: np.random.Generator,
) -> float:

    s = x1_.len * lam / (tau_ + x1_.len * lam)
    fitted_sum = sum([x2_.len @ alp_ for alp_, x2_ in zip(alp_sub, x2_sub)])
    post_prec = tau0 + tau_ * np.sum(s)
    post_mean = (tau_ / post_prec) * s @ ((x1_.sum - fitted_sum) / np.where(x1_.len == 0, np.inf, x1_.len))
    post_sd = 1 / np.sqrt(post_prec)
    return float(ome.normal(post_mean, post_sd))


def update_coefs_single(
    x1_: SuffStat1,
    x2_sub: list[SuffStat2],
    alp0: float,
    alp_sub: list[FloatArr],
    tau_: float,
    lam: float,
    ome: np.random.Generator,
) -> FloatArr:

    fitted_sum = sum([x2_.len @ alp_ for alp_, x2_ in zip(alp_sub, x2_sub)])
    post_prec = x1_.len * lam + tau_
    post_mean = (x1_.sum - x1_.len * alp0 - fitted_sum) * lam / post_prec
    post_sd = 1 / np.sqrt(post_prec)
    return ome.normal(post_mean, post_sd)


def update_resid_precision(
    x0: SuffStat0,
    x1: list[SuffStat1],
    x2: dict[tuple[int, int], SuffStat2],
    alp0: float,
    alp: list[FloatArr],
    prior_n: float,
    prior_est: float,
    ome: np.random.Generator,
) -> float:

    o0 = x0.sum2 + x0.len * alp0 ** 2 - 2 * alp0 * x0.sum
    o1 = sum([np.sum(x1_.len * np.square(alp_) + 2 * alp_ * (x1_.len * alp0 - x1_.sum)) for alp_, x1_ in zip(alp, x1)])
    o2 = 2 * sum([alp[k_[0]] @ x2_.len @ alp[k_[1]] for k_, x2_ in x2.items()])
    ssq_resid = o0 + o1 + o2
    post_n = prior_n + x0.len
    post_est = (prior_n * prior_est + ssq_resid) / post_n
    return ome.gamma(post_n / 2, 2 / (post_n * post_est))


def reduce_data(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    i: IntArr,
) -> tuple[SuffStat0, list[SuffStat1], dict[tuple[int, int], SuffStat2]]:

    x0 = SuffStat0(sum(n), sum(y1), sum(y2))
    x1 = [SuffStat1(*v.reindex(index=np.arange(v.index.max() + 1)).fillna(0).T.values)
          for v in marginalize_table([n, y1], i, 1).values()]
    x2 = {k: SuffStat2(*[csr_matrix((v[c].values, v.index.to_frame().T.values)) for c in v])
          for k, v in marginalize_table([n, y1], i, 2).items()}
    return x0, x1, x2


def marginalize_table(
    stats: list[FloatArr],
    i: IntArr,
    order: int,
) -> dict[tuple[int, ...], pd.DataFrame]:

    data = pd.DataFrame(np.array(stats).T, index=pd.MultiIndex.from_arrays(i.T)).sort_index()
    return {c: data.groupby(level=c).sum()
            for c in it.combinations(list(range(data.index.to_frame().shape[1])), order)}

from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.olm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    prior_n_tau: FloatArr = None,
    prior_est_tau: FloatArr = None,
    prior_n_lam: float = 1,
    init: gibbs.ParamSpace = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_posterior(y, j, i, eval_cdf, prior_n_tau, prior_est_tau, prior_n_lam, init, ome)


def eval_cdf(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    cdf = np.exp(-np.exp(-eta))
    d_cdf = np.exp(-eta - np.exp(-eta))
    d2_cdf = np.exp(-eta - np.exp(-eta)) * (-1 - np.exp(-eta))
    return np.where(np.isnan(cdf), 0, cdf), np.where(np.isnan(d_cdf), 0, d_cdf), np.where(np.isnan(d2_cdf), 0, d2_cdf)

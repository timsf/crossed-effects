from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from xfx.olm import gibbs


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: FloatArr | None = None,
    prior_n_lam: float = 1,
    init: gibbs.ParamSpace | None = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_posterior(y, j, i, eval_cdf, prior_n_tau, prior_est_tau, prior_n_lam, init, ome)


def eval_cdf(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    cdf = expit(eta)
    d_cdf = cdf * (1 - cdf)
    d2_cdf = d_cdf * (1 - 2 * cdf)
    return cdf, d_cdf, d2_cdf

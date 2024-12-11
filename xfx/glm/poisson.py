from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    prior_n_tau: FloatArr = None,
    prior_est_tau: FloatArr = None,
    tau0: float = 0,
    init: gibbs.ParamSpace = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_reglr_posterior(y, n, j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    mu = np.exp(eta)
    mu = np.where(np.isinf(mu), np.nan, mu)
    return mu, mu, mu

from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from xfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: float = 0,
    prior_n_tau: FloatArr = None,
    prior_est_tau: FloatArr = None,
    init: gibbs.ParamSpace = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_reglr_posterior(y, n, j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    mu = expit(eta)
    return np.logaddexp(0, eta), mu, mu * (1 - mu)

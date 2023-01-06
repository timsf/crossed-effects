from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from xfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    prior_n_tau: FloatArr = None,
    prior_est_tau: FloatArr = None,
    init: Tuple[List[FloatArr], FloatArr] = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[Tuple[List[FloatArr], FloatArr]]:

    return gibbs.sample_posterior(y, n, j, i, eval_part, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> Tuple[FloatArr, FloatArr, FloatArr]:

    mu = expit(eta)
    return np.logaddexp(0, eta), mu, mu * (1 - mu)

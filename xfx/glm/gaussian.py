from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt

from xfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    prior_n_tau: FloatArr = None,
    prior_est_tau: FloatArr = None,
    prior_n_phi: float = 1,
    prior_est_phi: float = 1,
    init: Tuple[List[FloatArr], FloatArr, float] = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[Tuple[List[FloatArr], FloatArr, float]]:

    return gibbs.sample_disp_posterior(
        y1, y2, n, j, i, eval_part, eval_base,
        prior_n_tau, prior_est_tau, prior_n_phi, prior_est_phi, init, collapse, ome)


def eval_part(eta: FloatArr) -> Tuple[FloatArr, FloatArr, FloatArr]:

    return np.square(eta) / 2, eta, np.ones(len(eta))


def eval_base(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    phi: float,
) -> Tuple[float, float, float]:

    log_g = - sum(n) * np.log(2 * np.pi * phi) / 2 - sum(y2) / (2 * phi)
    d_log_g = - sum(n) / (2 * phi) + sum(y2) / (2 * phi ** 2)
    d2_log_g = sum(n) / (2 * phi ** 2) - sum(y2) / phi ** 3
    return log_g, d_log_g, d2_log_g

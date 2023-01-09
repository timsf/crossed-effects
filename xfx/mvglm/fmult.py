from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from xfx.mvglm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: List[FloatArr] = None,
    init: Tuple[List[FloatArr], List[FloatArr]] = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[Tuple[List[FloatArr], List[FloatArr]]]:

    return gibbs.sample_posterior(y, np.sum(y, 1), j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> Tuple[FloatArr, FloatArr]:

    log_g = np.float_(logsumexp(eta, 1))
    d_log_g = np.exp(eta - log_g[:, np.newaxis])
    return log_g, d_log_g

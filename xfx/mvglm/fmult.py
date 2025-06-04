from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from xfx.mvglm import gibbs


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: FloatArr | None = None,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: list[FloatArr] | None = None,
    init: gibbs.ParamSpace | None = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_posterior(y, np.sum(y, 1), j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr]: 

    log_g = logsumexp(eta, 1)
    d_log_g = np.exp(eta - log_g[:, np.newaxis])
    return log_g, d_log_g

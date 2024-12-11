from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.mixmvglm import gibbs
from xfx.mvglm import fmult


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: list[FloatArr] = None,
    init: gibbs.ParamSpace = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_posterior(y, np.sum(y, 1), j, i, x, fmult.eval_part, eval_part2, tau0, prior_n_tau, prior_est_tau, init, ome)


def eval_part2(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    log_g, d_log_g = fmult.eval_part(eta)
    d2_log_g = np.array([np.diag(d_log_g_) for d_log_g_ in d_log_g]) - np.einsum('ij,ik->ijk', d_log_g, d_log_g)
    return log_g, d_log_g, d2_log_g

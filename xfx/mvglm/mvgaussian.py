from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.mvglm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: list[FloatArr] = None,
    init: gibbs.ParamSpace = None,
    collapse: bool = True,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

        return gibbs.sample_posterior(y, n, j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr]:

    return np.sum(np.square(eta), 1) / 2, eta

from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.mixglm import gibbs
from xfx.glm import poisson


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    tau0: float = 0,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: FloatArr | None = None,
    init: gibbs.ParamSpace | None = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_reglr_posterior(y, n, j, i, x, poisson.eval_part, tau0, prior_n_tau, prior_est_tau, init, ome)

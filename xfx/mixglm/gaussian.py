from typing import Iterator

import numpy as np
import numpy.typing as npt

from xfx.mixglm import gibbs
from xfx.glm import gaussian


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def sample_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    x: FloatArr,
    tau0: float = 0,
    prior_n_tau: FloatArr | None = None,
    prior_est_tau: FloatArr | None = None,
    prior_n_phi: float = 1,
    prior_est_phi: float = 1,
    init: gibbs.DispParamSpace | None = None,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.DispParamSpace]:

    return gibbs.sample_disp_posterior(
        y1, y2, n, j, i, x, gaussian.eval_part, gaussian.eval_base, tau0,
        prior_n_tau, prior_est_tau, prior_n_phi, prior_est_phi, init, ome)

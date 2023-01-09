from typing import List

import numpy as np
import numpy.typing as npt
from scipy.stats import wishart

from xfx.misc.linalg import sherman_morrison_update


FloatArr = npt.NDArray[np.float_]


def update_factor_precision(
    j: FloatArr,
    alp: List[FloatArr],
    prior_n: FloatArr,
    prior_est: List[FloatArr],
    ome: np.random.Generator,
) -> List[FloatArr]:

    post_n = prior_n + j
    post_est = [prior_est_ if np.isinf(prior_n_)
                    else post_n_ * sherman_morrison_update(prior_est_ / prior_n_, alp_, alp_)
                for prior_n_, prior_est_, post_n_, alp_ in zip(prior_n, prior_est, post_n, alp)]
    return [post_est_ if np.isinf(post_n_)
                else wishart.rvs(post_n_, post_est_ / post_n_, random_state=ome)
            for post_n_, post_est_ in zip(post_n, post_est)]

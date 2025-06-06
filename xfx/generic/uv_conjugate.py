import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def update_factor_precision(
    j: IntArr | FloatArr,
    alp: list[FloatArr],
    prior_n: FloatArr,
    prior_est: FloatArr,
    ome: np.random.Generator,
) -> FloatArr:

    post_n = prior_n + j
    post_est = np.where(
        np.isinf(prior_n), prior_est,
        post_n / (prior_n / prior_est + np.array([np.sum(np.square(alp_)) for alp_ in alp])))
    return np.where(np.isinf(post_n), post_est, ome.gamma(post_n / 2, (2 * post_est) / post_n))

from typing import List

import numpy as np


def update_factor_precision(j: np.ndarray, alp: List[np.ndarray], prior_n: np.ndarray, prior_est: np.ndarray, 
                            ome: np.random.Generator) -> np.ndarray:

    post_n = prior_n + j
    post_est = np.where(
        np.isinf(prior_n), prior_est,
        post_n / (prior_n / prior_est + np.array([np.sum(np.square(alp_)) for alp_ in alp])))
    return np.where(np.isinf(post_n), post_est, ome.gamma(post_n / 2, (2 * post_est) / post_n))

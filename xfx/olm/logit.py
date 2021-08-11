from typing import Iterator, List, Tuple

import numpy as np
from scipy.special import expit

from xfx.olm import gibbs


def sample_posterior(y: np.ndarray, j: np.ndarray, i: np.ndarray,
                     prior_n_tau: np.ndarray = None, prior_est_tau: np.ndarray = None,
                     prior_n_lam: float = 1, init: Tuple[List[np.ndarray], np.ndarray, np.ndarray] = None,
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:

    return gibbs.sample_posterior(y, j, i, eval_cdf, prior_n_tau, prior_est_tau, prior_n_lam, init, ome)


def eval_cdf(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cdf = expit(eta)
    d_cdf = cdf * (1 - cdf)
    d2_cdf = d_cdf * (1 - 2 * cdf)
    return cdf, d_cdf, d2_cdf

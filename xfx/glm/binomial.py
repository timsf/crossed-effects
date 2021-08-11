from typing import Iterator, List, Tuple

import numpy as np
from scipy.special import expit

from xfx.glm import gibbs


def sample_posterior(y: np.ndarray, n: np.ndarray, j: np.ndarray, i: np.ndarray,
                     prior_n_tau: np.ndarray = None, prior_est_tau: np.ndarray = None,
                     init: Tuple[List[np.ndarray], np.ndarray] = None,
                     collapse: bool = True, ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], np.ndarray]]:

    return gibbs.sample_posterior(y, n, j, i, eval_part, prior_n_tau, prior_est_tau, init, collapse, ome)


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    mu = expit(eta)
    return np.logaddexp(0, eta), mu, mu * (1 - mu)

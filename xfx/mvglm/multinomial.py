from typing import Iterator, List, Tuple

import numpy as np
from scipy.special import logsumexp

from xfx.mvglm import gibbs


def sample_posterior(y: np.ndarray, j: np.ndarray, i: np.ndarray,
                     tau0: np.ndarray = None, prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None,
                     init: Tuple[List[np.ndarray], List[np.ndarray]] = None, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:

    return gibbs.sample_posterior(y[:, :-1], np.sum(y, 1), j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, ome)


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    log_g = np.float_(logsumexp(np.hstack([eta, np.zeros([eta.shape[0], 1])]), 1))
    d_log_g = np.exp(eta - log_g[:, np.newaxis])
    return log_g, d_log_g

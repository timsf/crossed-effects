from typing import Iterator, List, Tuple

import numpy as np

from xfx.mvglm import gibbs


def sample_posterior(y: np.ndarray, n: np.ndarray, j: np.ndarray, i: np.ndarray,
                     tau0: np.ndarray = None, prior_n_tau: np.ndarray = None, prior_est_tau: [np.ndarray] = None,
                     init: (np.ndarray, [np.ndarray], [np.ndarray]) = None, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:

        return gibbs.sample_posterior(y, n, j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, ome)


def eval_part(eta: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    return np.sum(np.square(eta), 1) / 2, eta, np.repeat(np.identity(eta.shape[1])[np.newaxis], eta.shape[0], 0)

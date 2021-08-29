from typing import Iterator, List, Tuple

import numpy as np

from xfx.glm import gibbs


def sample_posterior(y1: np.ndarray, y2: np.ndarray, n: np.ndarray, j: np.ndarray, i: np.ndarray,
                     prior_n_tau: np.ndarray = None, prior_est_tau: np.ndarray = None,
                     prior_n_phi: float = 1, prior_est_phi: float = 1,
                     init: Tuple[List[np.ndarray], np.ndarray, float] = None,
                     collapse: bool = True, ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], np.ndarray, float]]:

    return gibbs.sample_disp_posterior(y1, y2, n, j, i, eval_part, eval_base,
                                       prior_n_tau, prior_est_tau, prior_n_phi, prior_est_phi, init, collapse, ome)


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    return np.square(eta) / 2, eta, np.ones(len(eta))


def eval_base(y1: np.ndarray, y2: np.ndarray, n: np.ndarray, phi: float) -> Tuple[float, float, float]:

    log_g = - sum(n) * np.log(2 * np.pi * phi) / 2 - sum(y2) / (2 * phi)
    d_log_g = - sum(n) / (2 * phi) + sum(y2) / (2 * phi ** 2)
    d2_log_g = sum(n) / (2 * phi ** 2) - sum(y2) / phi ** 3
    return log_g, d_log_g, d2_log_g
    
from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt

from xfx.mvglm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(y: FloatArr, n: FloatArr, j: IntArr, i: IntArr,
                     tau0: FloatArr = None, prior_n_tau: FloatArr = None, prior_est_tau: List[FloatArr] = None,
                     init: Tuple[List[FloatArr], List[FloatArr]] = None, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[FloatArr], List[FloatArr]]]:

        return gibbs.sample_posterior(y, n, j, i, eval_part, tau0, prior_n_tau, prior_est_tau, init, ome)


def eval_part(eta: FloatArr) -> Tuple[FloatArr, FloatArr]:

    return np.sum(np.square(eta), 1) / 2, eta

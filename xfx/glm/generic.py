from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]
PartFunc = Callable[[FloatArr], tuple[FloatArr, FloatArr, FloatArr]]
BaseFunc = Callable[[FloatArr, FloatArr, FloatArr, float], tuple[float, float, float]]


def eval_densities(
    y1: FloatArr, 
    n: FloatArr, 
    eta: FloatArr, 
    eval_part: PartFunc
) -> tuple[FloatArr, FloatArr, FloatArr]:

    part, d_part, d2_part = eval_part(eta)
    log_f = y1 * eta - n * part
    d_log_f = y1 - n * d_part
    d2_log_f = - n * d2_part
    return log_f, d_log_f, d2_log_f


def update_dispersion(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    eta: FloatArr,
    phi: float,
    eval_part: PartFunc,
    eval_base: BaseFunc,
    prior_n: float,
    prior_est: float,
    ome: np.random.Generator,
) -> float:

    def eval_log_p(phi_: float, log_v: float) -> tuple[float, float, float]:
        log_g, d_log_g, d2_log_g = eval_base(y1, y2, n, phi_)
        log_prior, d_log_prior, d2_log_prior = eval_logprior_phi(phi_, prior_n, prior_est)
        log_p = log_prior + log_g + log_p_nil / phi_ - log_v
        d_log_p = d_log_prior + d_log_g - log_p_nil / phi_ ** 2
        d2_log_p = d2_log_prior + d2_log_g + 2 * log_p_nil / phi_ ** 3
        return log_p, d_log_p, d2_log_p

    def brace(right: bool) -> float:
        sgn = 1 if right else -1
        width = 1
        while True:
            edge = phi * 2 ** (sgn * width)
            log_p, d_log_p, _ = eval_log_p(edge, log_u)
            if log_p < 0 and sgn * d_log_p < 0:
                return edge
            width += 1
    
    log_p_nil = np.sum(eval_densities(y1, n, eta, eval_part)[0]).item()
    log_u = eval_log_p(phi, 0)[0] - ome.exponential()
    lb = root_scalar(eval_log_p, (log_u,), bracket=(brace(False), phi), fprime=True, fprime2=True).root
    ub = root_scalar(eval_log_p, (log_u,), bracket=(phi, brace(True)), fprime=True, fprime2=True).root
    return ome.uniform(lb, ub)


def eval_logprior_phi(phi: float, prior_n: float, prior_est: float) -> tuple[float, float, float]:

    return -(prior_n / 2 + 1) * np.log(phi) - prior_n * prior_est / (2 * phi), \
           -(prior_n / 2 + 1) / phi + prior_n * prior_est / (2 * phi ** 2), \
           (prior_n / 2 + 1) / phi ** 2 - prior_n * prior_est / phi ** 3

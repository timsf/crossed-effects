from typing import Callable

import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
PartFunc = Callable[[FloatArr], tuple[FloatArr, FloatArr]]


def eval_densities(
    y: FloatArr, 
    n: FloatArr, 
    eta: FloatArr, 
    eval_part: PartFunc
) -> tuple[FloatArr, FloatArr, FloatArr]:
    
    part, d_part = eval_part(eta)
    log_f = np.sum(y * eta, 1) - n * part
    d_log_f = y - n[:, np.newaxis] * d_part
    return log_f, d_log_f

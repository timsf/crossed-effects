from typing import Callable

import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]
PartFunc2 = Callable[[FloatArr], tuple[FloatArr, FloatArr, FloatArr]]


def eval_densities2(
    y: FloatArr, 
    n: FloatArr, 
    eta: FloatArr, 
    eval_part: PartFunc2
) -> tuple[FloatArr, FloatArr, FloatArr]:
    
    part, d_part, d2_part = eval_part(eta)
    log_f = np.sum(y * eta, 1) - n * part
    d_log_f = y - n[:, np.newaxis] * d_part
    d2_log_f = - n[:, np.newaxis, np.newaxis] * d2_part
    return log_f, d_log_f, d2_log_f

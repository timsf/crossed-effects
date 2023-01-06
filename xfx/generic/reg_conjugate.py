from typing import Tuple

import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def est_gls(
    y: FloatArr,
    x: FloatArr,
    w: FloatArr,
) -> Tuple[FloatArr, FloatArr]:

    qN = x.T @ w @ x
    mN = np.linalg.solve(qN, x.T @ w @ y)
    return mN, qN

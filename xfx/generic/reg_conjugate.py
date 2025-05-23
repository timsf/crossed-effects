import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def est_gls(
    y: FloatArr,
    x: FloatArr,
    w: FloatArr,
    tau: FloatArr,
) -> tuple[FloatArr, FloatArr]:

    qN = x.T @ w @ x + tau
    mN = np.linalg.solve(qN, x.T @ w @ y)
    return mN, qN

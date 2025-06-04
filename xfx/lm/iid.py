import numpy as np
import numpy.typing as npt


IntArr = npt.NDArray[np.integer]
FloatArr = npt.NDArray[np.floating]


def update_intercept(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau: FloatArr,
    lam: float,
) -> tuple[float, float]:

    s = np.diag(1 / (lam * n))
    for k_ in range(i.shape[1]):
        for j_ in np.unique(i[:, k_]):
            s[np.ix_(i[:, k_] == j_, i[:, k_] == j_)] += 1 / tau[k_]

    post_var = float((n @ s @ n) / np.sum(n)) ** 2
    post_mean = float(np.sum(y1) / np.sum(n))

    post_var = float(n @ s @ n) / sum(n) ** 2
    post_mean = sum(y1) / sum(n)
    return post_mean, post_var


def update_coefs(
    y1: FloatArr,
    n: FloatArr,
    j: IntArr,
    i: IntArr,
    tau: FloatArr,
    lam: float,
) -> tuple[FloatArr, FloatArr]:

    s11 = np.diag(1 / np.repeat(tau, j))
    s22 = np.diag(1 / (lam * n))
    s12 = np.zeros((int(np.sum(j)), i.shape[0]))

    for k_ in range(len(j)):
        for j_ in range(j[k_]):
            s22[np.ix_(i[:, k_] == j_, i[:, k_] == j_)] += 1 / tau[k_]
            s12[j_ + np.sum(j[:k_]), i[:, k_] == j_] += 1 / tau[k_]

    t22 = np.linalg.inv(s22)
    m_mu, s_mu = update_intercept(y1, n, j, i, tau, lam)
    return s12 @ t22 @ ((y1 / n) - m_mu), \
           s11 - s12 @ t22 @ s12.T + s12 @ t22 @ np.ones((i.shape[0], i.shape[0])) @ t22 @ s12.T * s_mu

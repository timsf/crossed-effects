import numpy as np


def sherman_morrison_update(init: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:

    inv = init.copy()
    for a_, b_ in zip(a, b):
        inv -= np.outer(inv @ a_, inv.T @ b_) / (1 + b_ @ inv @ a_)
    return inv

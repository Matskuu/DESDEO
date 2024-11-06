
import math

import numpy as np


class GNIMBUSError(Exception):
    """Error raised when exceptions are encountered in Group NIMBUS."""

def decode(x: int | np.ndarray, k: int) -> np.ndarray:
    x_new = x - 1
    if not isinstance(x, np.ndarray):
        x_new = np.atleast_1d(x) - 1
        powers_of_3 = 3 ** np.arange(k - 1, -1, -1)
        return (x_new[:, None] // powers_of_3 % 3)[0]
    powers_of_3 = 3 ** np.arange(k - 1, -1, -1)
    return x_new[:, None] // powers_of_3 % 3

def encode(v: np.ndarray, k: int) -> int | np.ndarray:
    return v@(3 ** np.arange(k - 1, -1, -1)) - 1

def work_dom(x1: int | np.ndarray, x2: int | np.ndarray, k: int) -> bool  | list[bool]:
    # here we assume x1 and x2 are the same type and if arrays, same length as well
    if not isinstance(x1, np.ndarray):
        return np.all(decode(np.atleast_1d(x1), k) >= decode(np.atleast_1d(x2), k), axis=1)
    return np.all(decode(x1, k) >= decode(x2, k), axis=1)

def test_dom(k: int) -> list[bool]:
    size = 3 ** k
    indices = np.arange(1, size + 1)
    ret = np.fromfunction(
        lambda i, j: work_dom(indices[i].flatten(), indices[j].flatten(), k), (size, size), dtype=int
    )
    if ret.ndim == 1:
        ret = np.reshape(ret, (size, size))
        np.fill_diagonal(ret, False)
        return ret
    np.fill_diagonal(ret, False)
    return ret

def work_swap(
    i: int | np.ndarray,
    j: int | np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray
) -> bool | list[bool]:
    if isinstance(i, int) and isinstance(j, int):
        cond1 = np.logical_and(a[i] == ref[i], (a[i] == b[j]))
        cond2 = np.logical_and(a[j] == ref[j], (a[j] == b[i]))

        a1 = np.full((1, np.shape(a)[0]), a)
        b1 = np.full((1, np.shape(b)[0]), b)

        a1[np.arange(a1.shape[0]), i] = 9
        a1[np.arange(a1.shape[0]), j] = 9
        b1[np.arange(b1.shape[0]), i] = 9
        b1[np.arange(b1.shape[0]), j] = 9

        cond3 = np.all(a1 == b1)
        return np.logical_and.reduce((cond1, cond2, cond3, i != j))

    # i and j are index vectors for a, b and ref
    cond1 = np.logical_and(a[i] == ref[i], (a[i] == b[j]))
    cond2 = np.logical_and(a[j] == ref[j], (a[j] == b[i]))

    a1 = np.full((len(i), np.shape(a)[0]), a)
    b1 = np.full((len(i), np.shape(b)[0]), b)

    a1[np.arange(a1.shape[0]), i] = 9
    a1[np.arange(a1.shape[0]), j] = 9
    b1[np.arange(b1.shape[0]), i] = 9
    b1[np.arange(b1.shape[0]), j] = 9

    cond3 = np.all(a1 == b1, axis=1)
    return np.logical_and.reduce((cond1, cond2, cond3, i != j))

def test_swap(ref: list):
    k = len(ref)
    # number of classification vectors? change vectors?
    n_cv = int(math.pow(3, k))
    m = np.full((n_cv, n_cv), np.nan, dtype=np.bool)
    indices = np.arange(0, k-1)
    print(indices)

    for i in range(n_cv):
        decoded_i = decode(i, k)
        for j in range(n_cv):
            if i != j:
                decoded_j = decode(j, k)
                result = False
                for x in range(k):
                    for y in range(k):
                        if work_swap(x, y, decoded_i, decoded_j, ref):
                            result = True
                            break
                    if result:
                        break
                m[i][j] = result

    np.fill_diagonal(m, False)
    return

if __name__ == "__main__":
    #print(decode(np.array([1,2,0]),3))
    #print(decode(0,3))
    #print(encode(decode(np.array([1,2]),3), 3))
    #print(encode(decode(0,3), 3))
    #print(work_dom(0, 2, 3))
    #print(work_dom(np.array([1,2]), np.array([0,2]), 3))
    #print(test_dom(3))
    #print(work_swap(np.array([0,1]), np.array([1,2]), np.array([0, 1, 2]), np.array([0, 0, 1]), np.array([0, 1, 2])))
    print(test_swap([0,1,2]))



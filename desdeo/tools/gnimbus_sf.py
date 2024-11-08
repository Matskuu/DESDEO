
import math

import numpy as np


class GNIMBUSError(Exception):
    """Error raised when exceptions are encountered in Group NIMBUS."""

def decode(x: int | np.ndarray, n_obj: int) -> np.ndarray:
    x_new = x - 1
    if not isinstance(x, np.ndarray):
        x_new = np.atleast_1d(x) - 1
        powers_of_3 = 3 ** np.arange(n_obj - 1, -1, -1)
        return (x_new[:, None] // powers_of_3 % 3)[0]
    powers_of_3 = 3 ** np.arange(n_obj - 1, -1, -1)
    # the following acts as the outer function from R (perform an operation "//" for each element of an array)
    return x_new[:, None] // powers_of_3 % 3

def encode(v: np.ndarray, n_obj: int) -> int | np.ndarray:
    return v@(3 ** np.arange(n_obj - 1, -1, -1)) - 1

def work_dom(x1: int | np.ndarray, x2: int | np.ndarray, n_obj: int) -> bool  | list[bool]:
    # here we assume x1 and x2 are the same type and if arrays, same length as well
    if not isinstance(x1, np.ndarray):
        return np.all(decode(np.atleast_1d(x1), n_obj) >= decode(np.atleast_1d(x2), n_obj), axis=1)
    return np.all(decode(x1, n_obj) >= decode(x2, n_obj), axis=1)

def test_dom(n_obj: int) -> list[bool]:
    size = 3 ** n_obj
    indices = np.arange(1, size + 1)
    ret = np.fromfunction(
        lambda i, j: work_dom(indices[i].flatten(), indices[j].flatten(), n_obj), (size, size), dtype=int
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
    n_obj = len(ref)
    # number of classification vectors? change vectors?
    n_cv = int(math.pow(3, n_obj))
    m = np.full((n_cv, n_cv), np.nan, dtype=np.bool)

    for i in range(n_cv):
        decoded_i = decode(i + 1, n_obj) # i + 1 to match the indexing
        for j in range(n_cv):
            if i != j:
                decoded_j = decode(j + 1, n_obj) # j + 1 to match the indexing
                result = False
                for x in range(n_obj):
                    for y in range(n_obj):
                        if work_swap(x, y, decoded_i, decoded_j, ref):
                            result = True
                            break
                    if result:
                        break
                m[i][j] = result

    np.fill_diagonal(m, False)
    return m

def test_k_ratio(kr: float, n_obj: int):
    acv = decode(np.arange(1, 3 ** n_obj + 1), n_obj)
    n_imp = np.sum(acv == 2, axis=1)
    # the following acts as the outer function from R (perform an operation "/" for each element of an array)
    rel = n_imp[:, None] / n_imp
    rel = rel >= kr
    rel[np.isnan(rel)] = False
    return rel

def make_ranks(rel):
    # Ensure the matrix is square
    if rel.shape[0] != rel.shape[1]:
        raise ValueError("Error in makeRanks: rel must be a square matrix")

    # Check for NaN values
    if np.any(np.isnan(rel)):
        raise ValueError("Error in makeRanks: relation contains NaN")

    # Initialize rank array with NaN (unranked elements)
    rank = np.full(rel.shape[0], np.nan)

    # Initialize rank counter
    r_counter = 0

    # Main loop
    while not np.all(~rel.any(axis=1)):  # stop when all arcs are removed (no True values in rows)
        # Find all elements which are at the bottom (row sums are 0) and not yet ranked
        ind = np.where((np.sum(rel, axis=1) == 0) & np.isnan(rank))[0]
        #print(r_counter)
        #print(np.sum(rel, axis=1))
        rank[ind] = r_counter  # Assign rank
        r_counter += 1  # Increase rank counter

        # Drop used arcs (set the respective rows to False)
        rel[:, ind] = False

    # The last ones get the best rank
    ind = np.where((np.sum(rel, axis=1) == 0) & np.isnan(rank))[0]
    rank[ind] = r_counter
    return rank

def nd_compromise(comp, ranks):
    h = np.zeros((len(comp), len(comp)), dtype=bool)
    for i in range(len(comp)):
        for j in range(len(comp)):
            if i != j:
                h[i][j] = np.all(ranks[comp[i], :] >= ranks[comp[j], :]) and np.any(ranks[comp[i], :] != ranks[comp[j], :])
    nd = np.sum(h, axis=0) == 0
    return np.array([comp[i] for i, val in enumerate(nd) if val])

def main(refs, kr=None, print_intermediate=False):
    if refs.ndim == 1:
        refs = refs[np.newaxis, :]

    n_obj = refs.shape[1]

    # Test if change vectors are meaningful
    if np.any(np.sum(refs == 0, axis=1) == 0):
        raise ValueError("Each member must specify at least one objective to worsen")
    if np.any(np.sum(refs == 2, axis=1) == 0):
        raise ValueError("Each member must specify at least one objective to improve")

    rel = test_dom(n_obj)

    if print_intermediate:
        print("Dominance relation")
        print(rel)

    if kr is not None:
        h = test_k_ratio(kr, n_obj)
        rel = np.logical_or(rel, h)
        if print_intermediate:
            print("Relation from k-ratio")
            print(h)
    #print(rel)
    ranks = np.zeros((rel.shape[0],0), bool)
    for i in range(refs.shape[0]):
        h = test_swap(refs[i, :])
        r1 = np.logical_or(rel, h)
        if print_intermediate:
            print(f"Specific relation for member {i}")
        ranks = np.column_stack((ranks, make_ranks(r1) if ranks is not None else make_ranks(r1)))

    acv = decode(np.arange(1, 3 ** n_obj + 1), n_obj)
    infeas = np.sum(acv == 0, axis=1) == 0
    ranks[infeas, :] = -1

    minrank = np.min(ranks, axis=1)
    compromise = np.where(minrank == np.max(minrank))[0]
    compromise = nd_compromise(compromise, ranks)

    return {
        "ranks": ranks,
        "compromise": decode(compromise + 1, n_obj),
        "cranks": ranks[compromise, :]
    }

if __name__ == "__main__":
    #print(decode(np.array([1,2,0]),3))
    #print(decode(0,3))
    #print(encode(decode(np.array([1,2]),3), 3))
    #print(encode(decode(0,3), 3))
    #print(work_dom(0, 2, 3))
    #print(work_dom(np.array([1,2]), np.array([0,2]), 3))
    #print(test_dom(3))
    #print(work_swap(np.array([0,1]), np.array([1,2]), np.array([0, 1, 2]), np.array([0, 0, 1]), np.array([0, 1, 2])))
    #print(test_swap([2,0,1]))
    #print(test_k_ratio(1, 2))
    #rel = test_swap(np.array([1,2]))
    #print(make_ranks(rel))
    example = np.array([[2, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 1]])
    result = main(example)
    print(result)

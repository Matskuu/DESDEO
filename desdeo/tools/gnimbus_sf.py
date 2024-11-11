"""Aggregation of change vectors for GNIMBUS.

Change vectors are encoded as vectors of integer numbers between 0 and 2:

    0: objective can be worsened
    1: objective can stay the same
    2: objective can be improved.
"""

import math

import numpy as np


class GNIMBUSError(Exception):
    """Error raised when exceptions are encountered in Group NIMBUS."""

def decode(x: int | np.ndarray, n_obj: int) -> np.ndarray:
    """Decode a number (1-based index) into its change vector representation.

    Args:
        x (int | np.ndarray): integer or numpy array of indices
        n_obj (int): number of objectives (size of the change vector)

    Returns:
        np.ndarray: change vector representation as numpy array
    """
    # find a change vector from its number
    # note that numbers are 1 plus the base 3 coding
    x_new = x - 1
    if not isinstance(x, np.ndarray):
        x_new = np.atleast_1d(x) - 1
        powers_of_3 = 3 ** np.arange(n_obj - 1, -1, -1)
        return (x_new[:, None] // powers_of_3 % 3)[0]
    powers_of_3 = 3 ** np.arange(n_obj - 1, -1, -1)
    # the following acts as the outer function from R (perform an operation "//" for each element of an array)
    return x_new[:, None] // powers_of_3 % 3

def encode(v: np.ndarray, n_obj: int) -> int | np.ndarray:
    """Encode a change vector (matrix) into its index number.

    Args:
        v (np.ndarray): numpy array of change vectors (rows of 0, 1, 2)
        n_obj (int): number of objectives

    Returns:
        int | np.ndarray: encoded change vector as index
    """
    return v@(3 ** np.arange(n_obj - 1, -1, -1)) - 1

def work_dom(i: int | np.ndarray, j: int | np.ndarray, n_obj: int) -> bool  | list[bool]:
    """Work function for creating dominance relations.

    Args:
        i (int | np.ndarray): change vector index (or indices)
        j (int | np.ndarray): change vector index (or indices)
        n_obj (int): number of objectives

    Returns:
        bool  | list[bool]: dominance relations
    """
    if not isinstance(i, np.ndarray):
        return np.all(decode(np.atleast_1d(i), n_obj) >= decode(np.atleast_1d(j), n_obj), axis=1)
    return np.all(decode(i, n_obj) >= decode(j, n_obj), axis=1)

def test_dom(n_obj: int) -> np.ndarray:
    """Create dominance relation matrix.

    Args:
        n_obj (int): number of objectives

    Returns:
        np.ndarray: dominance relation matrix
    """
    size = 3 ** n_obj
    indices = np.arange(1, size + 1)
    ret = np.fromfunction(
        lambda i, j: work_dom(indices[i].flatten(), indices[j].flatten(), n_obj), (size, size), dtype=int
    )
    if ret.ndim == 1:
        ret = np.reshape(ret, (size, size))
    np.fill_diagonal(ret, False)
    return ret

def work_swap(
    i: int | np.ndarray,
    j: int | np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray
) -> bool | list[bool]:
    """Work function for the swap relation.

    Args:
        i (int | np.ndarray): index (or indices) or objectives
        j (int | np.ndarray): index (or indices) or objectives
        a (np.ndarray): change vector
        b (np.ndarray): change vector
        ref (np.ndarray): reference change vector

    Returns:
        bool | list[bool]: swap relations
    """
    # comparing two change vectors a and b
    # we look for pairs of objectives i and j that fulfill the conditions that
    # 1) a[i] and a[j] are the same as in the reference vector
    # 2) b[i] and b[j] are swapped from a
    # 3) all other elements are the same between a and b (that is the most tricky part)
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

def test_swap(ref: list) -> np.ndarray:
    """Create relation matrix from swaps.

    Args:
        ref (list): reference change vector

    Returns:
        np.ndarray: relation matrix from swaps
    """
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

def test_k_ratio(kr: float, n_obj: int) -> np.ndarray:
    """Test k-ratio relation.

    Args:
        kr (float): threshold ratio
        n_obj (int): number of objectives

    Returns:
        np.ndarray: k-ratio relations
    """
    # all change vectors (acv)
    acv = decode(np.arange(1, 3 ** n_obj + 1), n_obj)
    # number of improved objectives
    n_imp = np.sum(acv == 2, axis=1)
    # the following acts as the outer function from R (perform an operation "/" for each element of an array)
    rel = n_imp[:, None] / n_imp
    rel = rel >= kr
    rel[np.isnan(rel)] = False
    return rel

def make_ranks(rel: np.ndarray) -> np.ndarray:
    """Calculate ranks from the binary relation matrix.

    Args:
        rel (np.ndarray): binary relation matrix

    Raises:
        ValueError: if rel not a square matrix or rel contains NaN
        GNIMBUSError: if while loop infinite

    Returns:
        np.ndarray: array of ranks, higher ranks imply better alternatives
    """
    # Ensure the matrix is square
    if rel.shape[0] != rel.shape[1]:
        raise ValueError("Error in make_ranks: rel must be a square matrix")

    # Check for NaN values
    if np.any(np.isnan(rel)):
        raise ValueError("Error in make_ranks: relation contains NaN")

    # Initialize rank array with NaN (unranked elements)
    rank = np.full(rel.shape[0], np.nan)

    # Initialize rank counter
    r_counter = 0

    # Initialize prev_rank to get another stop condition (to avoid infinite loops)
    prev_rank = None

    # Main loop
    while not np.all(~rel.any(axis=1)): # stop when all arcs are removed (no True values in rows)
        # Find all elements which are at the bottom (row sums are 0) and not yet ranked
        ind = np.where((np.sum(rel, axis=1) == 0) & np.isnan(rank))[0]
        prev_rank = rank.copy()
        rank[ind] = r_counter  # Assign rank
        # Stop the loop if rank array stays the same (to avoid infinite loops)
        if np.array_equal(prev_rank, rank, equal_nan=True):
            raise GNIMBUSError("Error in making ranks.")
        r_counter += 1  # Increase rank counter
        # Drop used arcs (set the respective rows to False)
        rel[:, ind] = False

    # The last ones get the best rank
    ind = np.where((np.sum(rel, axis=1) == 0) & np.isnan(rank))[0]
    rank[ind] = r_counter
    return rank

def nd_compromise(comp: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Find non-dominated compromise alternatives.

    Args:
        comp (np.ndarray): indices of compromise alternatives
        ranks (np.ndarray): rank matrix

    Returns:
        np.ndarray: non-dominated compromise alternatives
    """
    h = np.zeros((len(comp), len(comp)), dtype=bool)
    for i in range(len(comp)):
        for j in range(len(comp)):
            if i != j:
                h[i][j] = np.all(ranks[comp[i], :] >= ranks[comp[j], :]) and np.any(ranks[comp[i], :] != ranks[comp[j], :])
    nd = np.sum(h, axis=0) == 0
    return np.array([comp[i] for i, val in enumerate(nd) if val])

def main(refs: np.ndarray, kr: float | None = None, print_intermediate: bool | None = False) -> dict:
    """Main function to find compromise change vector.

    Args:
        refs (np.ndarray): matrix of individual change vectors (rows)
        kr (float, optional): threshold for k-ratio. Defaults to None.
        print_intermediate (bool, optional): whether to print intermediate results. Defaults to False.

    Raises:
        ValueError: if change vectors are infeasible

    Returns:
        dict: dict with the ranks, compromise vectors and compromise vector ranks
    """
    if refs.ndim == 1:
        refs = refs[np.newaxis, :]

    n_obj = refs.shape[1]

    # Test if change vectors are meaningful
    if np.any(np.sum(refs == 0, axis=1) == 0):
        raise ValueError("Each member must specify at least one objective to worsen")
    if np.any(np.sum(refs == 2, axis=1) == 0):
        raise ValueError("Each member must specify at least one objective to improve")

    # dominance relation
    rel = test_dom(n_obj)

    if print_intermediate:
        print("Dominance relation")
        print(rel)

    # possible add ratio
    if kr is not None:
        h = test_k_ratio(kr, n_obj)
        rel = np.logical_or(rel, h)
        if print_intermediate:
            print("Relation from k-ratio")
            print(h)

    ranks = np.zeros((rel.shape[0],0), bool)
    for i in range(refs.shape[0]):
        h = test_swap(refs[i, :])
        r1 = np.logical_or(rel, h)
        if print_intermediate:
            print(f"Specific relation for member {i}")
        ranks = np.column_stack((ranks, make_ranks(r1) if ranks is not None else make_ranks(r1)))

    # all infeasible vectors (no decrease) receive rank -1
    acv = decode(np.arange(1, 3 ** n_obj + 1), n_obj)
    infeas = np.sum(acv == 0, axis=1) == 0
    ranks[infeas, :] = -1

    # find maxmin rank
    minrank = np.min(ranks, axis=1)
    compromise = np.where(minrank == np.max(minrank))[0]
    compromise = nd_compromise(compromise, ranks)

    return {
        "ranks": ranks,
        "compromise": decode(compromise + 1, n_obj),
        "cranks": ranks[compromise, :]
    }

if __name__ == "__main__":
    # Example Usage
    example = np.array([[2, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 1]])
    result = main(example)
    print(result)

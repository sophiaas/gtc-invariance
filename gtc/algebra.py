"""Algebraic computations."""

import numpy as np
from escnn.group import CyclicGroup, DihedralGroup


def _remove_redundant_indices(indices_set, group):
    """Remove redundant pairs of group indices for commutative groups.

    This uses the symmetries of the triple correlations that exist
    for commutative groups.

    Parameters
    ----------
    indices_set : set of tuples
        The (multi)-indices of the pairs of group elements.
    group : escnn.group.Group
        The group object.
        The group inverse and composition ("combination") are used.

    Returns
    -------
    out_set : set of tuples
        The indices of the pairs of group elements, with redundant pairs removed."""
    out_set = set()

    while indices_set:
        pair = indices_set.pop()
        out_set.add(pair)
        g1, g2 = pair

        g1_inv = group._inverse(g1)
        g2_inv = group._inverse(g2)

        g2_g1_inv = group._combine(g2, g1_inv)
        g1_g2_inv = group._combine(g1, g2_inv)

        to_remove = [
            (g1_inv, g2_g1_inv),
            (g2_g1_inv, g1_inv),
            (g2_inv, g1_g2_inv),
            (g1_g2_inv, g2_inv),
        ]

        for pair in to_remove:
            if pair in indices_set:
                indices_set.remove(pair)

    return out_set


def compute_non_redundant_tc_indices_cyclic(N=8):
    """Compute pairs of group elements indices for the TC of a cyclic group.

    Parameters
    ----------
    N : int
        Number of rotations included in C_N which discretizes SO(2).

    Returns
    -------
    indices_set : set of tuples
        The indices of the pairs of rotations to be used in the TC.
        One element index g is an integer in {0, ..., N-1}.
    """
    c_group = CyclicGroup(N=N)
    first_indices, second_indices = np.triu_indices(N)

    indices_set = set(
        (int(g1), int(g2)) for g1, g2 in zip(first_indices, second_indices)
    )
    nonredundant_indices = _remove_redundant_indices(indices_set, c_group)
    row_idx = np.zeros(len(nonredundant_indices), dtype=np.int)
    col_idx = np.zeros(len(nonredundant_indices), dtype=np.int)
    for i, idx in enumerate(nonredundant_indices):
        row_idx[i] = idx[0]
        col_idx[i] = idx[1]
    return row_idx, col_idx


def compute_non_redundant_tc_indices_dihedral(N=8):
    """Compute pairs of group elements indices for the TC of a dihedral group.

    The Dihedral group is denoted D_N.

    Since D_N is not commutative, we do not use the symmetries that exist for
    commutative groups.

    Parameters
    ----------
    N : int
        Integer defining the dihedral group D_N.
        We note that the order of the group, that is its size, is 2N.

    Returns
    -------
    indices_set : set of tuples
        The multi-indices of the pairs of group elements to be used in the TC.
        One multi-index is a tuple g = (m, r) where:
        - m is in {0, 1}: the reflection index: 0 for the identity, 1 for the reflection
        - r is in {0, ..., N-1}: the rotation index
    """
    d_group = DihedralGroup(N=N)
    first_indices, second_indices = np.triu_indices(d_group.order())

    first_multi_indices = []
    second_multi_indices = []
    for idx in first_indices:
        idx = int(idx)
        first_multi_indices.append((idx // N, idx % N))
    for idx in second_indices:
        idx = int(idx)
        second_multi_indices.append((idx // N, idx % N))

    indices_set = set(
        (g1, g2) for g1, g2 in zip(first_multi_indices, second_multi_indices)
    )
    
    row_idx = np.zeros(len(indices_set), dtype=np.int)
    col_idx = np.zeros(len(indices_set), dtype=np.int)
    for i, idx in enumerate(indices_set):
        import pdb; pdb.set_trace()
        row_idx[i] = idx[0]
        col_idx[i] = idx[1]
    return row_idx, col_idx

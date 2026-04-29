import itertools

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def _mode(site: int, spin: int) -> int:
    return 2 * site + spin

def _sector_basis(L: int, n_particles: int | None):
    n_modes = 2 * L
    if n_particles is None:
        return list(range(1 << n_modes))
    return [
        sum(1 << mode for mode in modes)
        for modes in itertools.combinations(range(n_modes), n_particles)
    ]

def _annihilate(state: int, mode: int):
    bit = 1 << mode
    if state & bit == 0:
        return None
    sign = -1 if (state & (bit - 1)).bit_count() % 2 else 1
    return state ^ bit, sign

def _create(state: int, mode: int):
    bit = 1 << mode
    if state & bit:
        return None
    sign = -1 if (state & (bit - 1)).bit_count() % 2 else 1
    return state | bit, sign

def _add_hop(rows, cols, vals, index, col, dst: int, src: int, coeff: float):
    out = _annihilate(col, src)
    if out is None:
        return
    state, sign1 = out
    out = _create(state, dst)
    if out is None:
        return
    row, sign2 = out
    rows.append(index[row])
    cols.append(index[col])
    vals.append(coeff * sign1 * sign2)

def hamil(L: int, t: float = 1., U: float = 4., n_particles: int | None = None):
    if L < 2:
        raise ValueError('L must be at least 2')
    if n_particles is not None and not 0 <= n_particles <= 2 * L:
        raise ValueError('n_particles must be between 0 and 2L')

    basis = _sector_basis(L, n_particles)
    index = {state: idx for idx, state in enumerate(basis)}
    rows = []
    cols = []
    vals = []

    for col_idx, state in enumerate(basis):
        diag = 0.
        for site in range(L):
            n_up = (state >> _mode(site, 0)) & 1
            n_dn = (state >> _mode(site, 1)) & 1
            diag += U * (n_up - .5) * (n_dn - .5) / L
        if diag != 0:
            rows.append(col_idx)
            cols.append(col_idx)
            vals.append(diag)

    for state in basis:
        for site in range(L):
            site_next = (site + 1) % L
            for spin in (0, 1):
                mode = _mode(site, spin)
                mode_next = _mode(site_next, spin)
                _add_hop(rows, cols, vals, index, state, mode, mode_next, -t/L)
                _add_hop(rows, cols, vals, index, state, mode_next, mode, -t/L)

    dim = len(basis)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=float)

def gs(L: int, t: float = 1., U: float = 4., n_particles: int | None = None, tol=1e-10, vec=False):
    H = hamil(L, t=t, U=U, n_particles=n_particles)
    if vec:
        vals, vecs = eigsh(H, k=1, which='SA', tol=tol)
        return float(vals[0]), vecs[:, 0]
    val = eigsh(H, k=1, which='SA', return_eigenvectors=False, tol=tol)[0]
    return float(val)

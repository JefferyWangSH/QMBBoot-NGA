import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

_sx = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_sy = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_id = sparse.identity(2, format='csr', dtype=complex)
_paulis = (_sx, _sy, _sz)


def kron(ops):
    ops = tuple(ops)
    if not ops:
        raise ValueError('ops must be non-empty')
    out = ops[0]
    for op in ops[1:]:
        out = sparse.kron(out, op, format='csr')
    return out


def hamil(L, J1=1., J2=0.):
    if L < 3:
        raise ValueError('L must be at least 3')

    dim = 1 << L
    H = sparse.csr_matrix((dim, dim), dtype=complex)

    for i in range(L):
        for pauli in _paulis:
            ops = [_id] * L
            ops[i] = pauli
            ops[(i + 1) % L] = pauli
            H += J1 * kron(ops)

            ops = [_id] * L
            ops[i] = pauli
            ops[(i + 2) % L] = pauli
            H += J2 * kron(ops)

    return H / L


def gs(L, J1=1., J2=0., tol=1e-10, vec=False):
    H = hamil(L, J1=J1, J2=J2)
    if vec:
        vals, vecs = eigsh(H, k=1, which='SA', tol=tol)
        return float(vals[0].real), vecs[:, 0]
    val = eigsh(H, k=1, which='SA', return_eigenvectors=False, tol=tol)[0]
    return float(val.real)

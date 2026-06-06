import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

_sx = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
_sz = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=float))
_id = sparse.identity(2, format='csr', dtype=float)

def kron(ops):
    ops = tuple(ops)
    if not ops:
        raise ValueError('ops must be non-empty')
    out = ops[0]
    for op in ops[1:]:
        out = sparse.kron(out, op, format='csr')
    return out

def hamil(L, J=1., h=1., hz=0.):
    if L < 2:
        raise ValueError('L must be at least 2')

    dim = 1 << L
    H = sparse.csr_matrix((dim, dim), dtype=float)

    for i in range(L):
        ops = [_id] * L
        ops[i] = _sx
        H += -h * kron(ops)

    for i in range(L):
        ops = [_id] * L
        ops[i] = _sz
        H += -hz * kron(ops)

    for i in range(L):
        ops = [_id] * L
        ops[i] = _sz
        ops[(i + 1) % L] = _sz
        H += -J * kron(ops)

    return H / L

def zz(L, vec, r):
    states = np.arange(1 << L, dtype=np.uint64)
    sites = np.arange(L, dtype=np.uint64)
    spins = 1. - 2. * ((states[:, None] >> sites) & 1)
    corr = (spins * np.roll(spins, -r, axis=1)).mean(axis=1)
    return float(np.dot(np.abs(vec) ** 2, corr))

def gs(L, J=1., h=1., hz=0., tol=1e-12, vec=False):
    H = hamil(L, J=J, h=h, hz=hz)
    if vec:
        vals, vecs = eigsh(H, k=1, which='SA', tol=tol)
        return float(vals[0]), vecs[:, 0]
    val = eigsh(H, k=1, which='SA', return_eigenvectors=False, tol=tol)[0]
    return float(val)

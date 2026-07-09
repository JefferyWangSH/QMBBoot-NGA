import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

class IsingED:
    def __init__(
        self,
        L: int,
        J: float = 1.,
        h: float = 1.,
        hz: float = 0.,
        dtype=np.float64,
        **basis_kwargs,
    ):
        self.L = L
        self.J = J
        self.h = h
        self.hz = hz
        self.dtype = dtype
        self.basis = spin_basis_1d(L, pauli=1, **basis_kwargs)

        x_field = [[-h / L, i] for i in range(L)]
        z_field = [[-hz / L, i] for i in range(L)]
        zz_coupling = [[-J / L, i, (i + 1) % L] for i in range(L)]
        self.H = hamiltonian(
            [
                ['x', x_field],
                ['z', z_field],
                ['zz', zz_coupling],
            ],
            [],
            basis=self.basis,
            dtype=self.dtype,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )

        self.energy = None
        self.vec = None

    def solve(self, tol=1e-12, ncv=None, maxiter=None, vec=True):
        eigsh_kwargs = {'k': 1, 'which': 'SA', 'tol': tol}
        if ncv is not None:
            eigsh_kwargs['ncv'] = ncv
        if maxiter is not None:
            eigsh_kwargs['maxiter'] = maxiter

        if vec:
            vals, vecs = self.H.eigsh(return_eigenvectors=True, **eigsh_kwargs)
            self.energy = float(vals[0])
            self.vec = vecs[:, 0]
        else:
            val = self.H.eigsh(return_eigenvectors=False, **eigsh_kwargs)[0]
            self.energy = float(val)
            self.vec = None
        return self.energy

    def _expt(self, static, vec=None):
        if vec is None and self.vec is None:
            raise ValueError('solve with vec=True first, or pass vec explicitly')
        op = hamiltonian(
            static,
            [],
            basis=self.basis,
            dtype=self.dtype,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        return float(op.expt_value(vec if vec is not None else self.vec).real)

    def zz(self, r: int, vec=None):
        terms = [[1. / self.L, i, (i + r) % self.L] for i in range(self.L)]
        return self._expt([['zz', terms]], vec=vec)


if __name__ == '__main__':
    model = IsingED(L=16, J=1., h=1., hz=.1)
    print(f'L: {model.L}')
    print(f'J: {model.J}')
    print(f'h: {model.h}')
    print(f'hz: {model.hz}')
    print(f'basis size: {model.basis.Ns}')
    print(f'energy: {model.solve():.12f}')
    print(f'zz: {[model.zz(r) for r in range(model.L // 2 + 1)]}')

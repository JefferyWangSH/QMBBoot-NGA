import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

class HeisenbergED:
    def __init__(
        self,
        L: int,
        J1: float = 1.,
        J2: float = 1.,
        dtype=np.float64,
        **basis_kwargs,
    ):
        self.L = L
        self.J1 = J1
        self.J2 = J2
        self.dtype = dtype
        self.basis = spin_basis_1d(L, pauli=1, **basis_kwargs)

        nn = [[.25 * J1 / L, i, (i + 1) % L] for i in range(L)]
        nnn = [[.25 * J2 / L, i, (i + 2) % L] for i in range(L)]
        self.H = hamiltonian(
            [
                ['xx', nn],
                ['yy', nn],
                ['zz', nn],
                ['xx', nnn],
                ['yy', nnn],
                ['zz', nnn],
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

    def szz(self, r: int, vec=None):
        terms = [[.25 / self.L, i, (i + r) % self.L] for i in range(self.L)]
        return self._expt([['zz', terms]], vec=vec)


if __name__ == '__main__':
    model = HeisenbergED(L=16, J1=1., J2=1.)
    print(f'L: {model.L}')
    print(f'J1: {model.J1}')
    print(f'J2: {model.J2}')
    print(f'basis size: {model.basis.Ns}')
    print(f'energy: {model.solve():.12f}')
    print(f'szz: {[model.szz(r) for r in range(model.L // 2 + 1)]}')

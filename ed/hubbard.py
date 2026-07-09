import numpy as np
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

class HubbardED:
    def __init__(
        self,
        L: int,
        Nf: tuple[int, int],
        t: float = 1.,
        U: float = 4.,
        dtype=np.float64,
        **basis_kwargs,
    ):
        self.L = L
        self.Nf = Nf
        self.t = t
        self.U = U
        self.dtype = dtype
        self.basis = spinful_fermion_basis_1d(L, Nf, **basis_kwargs)
        self.const = U / 4

        hop_right = [[-t / L, i, (i + 1) % L] for i in range(L)]
        hop_left = [[-t / L, (i + 1) % L, i] for i in range(L)]
        interaction = [[U / L, i, i] for i in range(L)]
        number_shift = [[-U / (2 * L), i] for i in range(L)]
        self.H = hamiltonian(
            [
                ['+-|', hop_right],
                ['+-|', hop_left],
                ['|+-', hop_right],
                ['|+-', hop_left],
                ['n|n', interaction],
                ['n|', number_shift],
                ['|n', number_shift],
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
            self.energy = float(vals[0] + self.const)
            self.vec = vecs[:, 0]
        else:
            val = self.H.eigsh(return_eigenvectors=False, **eigsh_kwargs)[0]
            self.energy = float(val + self.const)
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
        L = self.L
        up_up = [[.25 / L, i, (i + r) % L] for i in range(L)]
        down_down = [[.25 / L, i, (i + r) % L] for i in range(L)]
        up_down = [[-.25 / L, i, (i + r) % L] for i in range(L)]
        down_up = [[-.25 / L, (i + r) % L, i] for i in range(L)]
        return self._expt([
            ['nn|', up_up],
            ['|nn', down_down],
            ['n|n', up_down],
            ['n|n', down_up],
        ], vec=vec)

    def double_occ(self, vec=None):
        terms = [[1. / self.L, i, i] for i in range(self.L)]
        return self._expt([['n|n', terms]], vec=vec)


if __name__ == '__main__':
    model = HubbardED(L=10, Nf=(5, 5))
    # model = HubbardED(L=16, Nf=(7, 7), kblock=0)
    print(f'L: {model.L}')
    print(f'Nf: {model.Nf}')
    print(f't: {model.t}')
    print(f'U: {model.U}')
    print(f'basis size: {model.basis.Ns}')
    print(f'energy: {model.solve():.12f}')
    print(f'double_occ: {model.double_occ():.12f}')
    print(f'szz: {[model.szz(r) for r in range(model.L // 2 + 1)]}')

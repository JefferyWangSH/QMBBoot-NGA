import numpy as np
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian

class HubbardSquareED:
    def __init__(
        self,
        Lx: int,
        Ly: int,
        Nf: tuple[int, int],
        t: float = 1.,
        U: float = 4.,
        dtype=np.float64,
        kxblock=None,
        kyblock=None,
        **basis_kwargs,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.n_sites = Lx * Ly
        self.Nf = Nf
        self.t = t
        self.U = U
        self.dtype = dtype

        if kxblock is not None:
            basis_kwargs['kxblock'] = (self._trans_x(), kxblock)
        if kyblock is not None:
            basis_kwargs['kyblock'] = (self._trans_y(), kyblock)

        self.basis = spinful_fermion_basis_general(self.n_sites, Nf, **basis_kwargs)
        self.const = U / 4

        hop_forward = []
        hop_backward = []
        for x in range(Lx):
            for y in range(Ly):
                i = self._site2idx(x, y)
                for j in (self._site2idx((x + 1) % Lx, y), self._site2idx(x, (y + 1) % Ly)):
                    hop_forward.append([-t / self.n_sites, i, j])
                    hop_backward.append([-t / self.n_sites, j, i])

        interaction = [[U / self.n_sites, i, i] for i in range(self.n_sites)]
        number_shift = [[-U / (2 * self.n_sites), i] for i in range(self.n_sites)]
        self.H = hamiltonian(
            [
                ['+-|', hop_forward],
                ['+-|', hop_backward],
                ['|+-', hop_forward],
                ['|+-', hop_backward],
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

    def _site2idx(self, x: int, y: int):
        return (x % self.Lx) + (y % self.Ly) * self.Lx

    def _trans_x(self):
        return np.array([
            self._site2idx(i % self.Lx + 1, i // self.Lx)
            for i in range(self.n_sites)
        ], dtype=np.int32)

    def _trans_y(self):
        return np.array([
            self._site2idx(i % self.Lx, i // self.Lx + 1)
            for i in range(self.n_sites)
        ], dtype=np.int32)

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

    def szz(self, dx: int, dy: int, vec=None):
        up_up = []
        down_down = []
        up_down = []
        down_up = []
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self._site2idx(x, y)
                j = self._site2idx(x + dx, y + dy)
                up_up.append([.25 / self.n_sites, i, j])
                down_down.append([.25 / self.n_sites, i, j])
                up_down.append([-.25 / self.n_sites, i, j])
                down_up.append([-.25 / self.n_sites, j, i])
        return self._expt([
            ['nn|', up_up],
            ['|nn', down_down],
            ['n|n', up_down],
            ['n|n', down_up],
        ], vec=vec)

    def double_occ(self, vec=None):
        terms = [[1. / self.n_sites, i, i] for i in range(self.n_sites)]
        return self._expt([['n|n', terms]], vec=vec)


if __name__ == '__main__':
    model = HubbardSquareED(Lx=2, Ly=2, Nf=(2, 2), kxblock=0, kyblock=0)
    print(f'Lx: {model.Lx}')
    print(f'Ly: {model.Ly}')
    print(f'Nf: {model.Nf}')
    print(f't: {model.t}')
    print(f'U: {model.U}')
    print(f'basis size: {model.basis.Ns}')
    print(f'energy: {model.solve():.12f}')
    print(f'double_occ: {model.double_occ():.12f}')
    print(f'szz: {[
        [model.szz(dx, dy) for dy in range(model.Ly)]
        for dx in range(model.Lx)
    ]}')

from dataclasses import dataclass
import numpy as np

from sdp import SDPSolver

@dataclass(slots=True)
class NGARecord:
    value: float | None = None
    status: str | None = None
    basis_reps: int | None = None
    psd_dims: list[int] | None = None
    n_vars: int | None = None
    null_count: int | None = None
    max_null_eigval: float | None = None
    to_drop: int | None = None
    to_grow: int | None = None
    net_growth: int | None = None


class NGARunner:
    '''
        Nullspace-guided adaptive (NGA) runner

        Required model interfaces:

            BasisRep.canon
            BasisRep.canon_rep

            Operator.__add__
            Operator.__rmul__
            Operator.commutator(op: Operator) -> Operator

            Compiler.block_reprs: list[list[BasisRep]]
            Compiler.psd_blocks: list[PSDConstraints]
            Compiler.vars: list
            Compiler.hamil_op: Operator
            Compiler.compile(basis_reprs: list[BasisRep]) -> None
            Compiler.sdp_data() -> SDPData
            Compiler.summary() -> dict
            Compiler.fourier(rep: BasisRep, block_idx: int) -> Operator
    '''
    def __init__(
        self,
        compiler,
        basis_reprs,
        required_basis_reprs,
        solver_backend='MOSEK',
        solver_kwargs=None,
        null_tol=1e-9,
        descendant_tol=1e-5,
        max_drop_leverage=1e-2,
        min_net_growth_per_step=1,
        max_net_growth_per_step=4,
        max_drop_per_step=8,
    ):
        self.compiler = compiler

        # canonical basis reprs (duplication removed)
        self.basis_reprs = []
        reprs_seen = set()
        for rep in basis_reprs:
            rep = rep.canon_rep
            if rep.canon in reprs_seen:
                continue
            reprs_seen.add(rep.canon)
            self.basis_reprs.append(rep)

        self.basis_indices = {rep.canon: idx for idx, rep in enumerate(self.basis_reprs)}
        self.required_keys = {rep.canon for rep in required_basis_reprs}
        if self.required_keys - set(self.basis_indices):
            raise ValueError('required basis representatives must be included in basis_reprs')

        self.solver_backend = solver_backend
        self.solver_kwargs = solver_kwargs or {}
        self.null_tol = null_tol
        self.descendant_tol = descendant_tol
        self.max_drop_leverage = max_drop_leverage
        self.min_net_growth_per_step = min_net_growth_per_step
        self.max_net_growth_per_step = max_net_growth_per_step
        self.max_drop_per_step = max_drop_per_step

        self.solver = SDPSolver()
        self.record = NGARecord(basis_reps = len(self.basis_reprs))
        self.history = []

        self.psd_eigvals = []
        self.psd_eigvecs = []

        self.leverage = None

        # small indices have higher priority
        self.to_drop = []
        self.to_grow = []

        self.null_count = 0
        self.max_null_eigval = None

    def build(self):
        self.compiler.compile(self.basis_reprs)
        self.solver.build(self.compiler.sdp_data())
        self.record.psd_dims = [psd.dim for psd in self.compiler.psd_blocks]
        self.record.n_vars = len(self.compiler.vars)

    def solve(self):
        value = self.solver.solve(backend=self.solver_backend, **self.solver_kwargs)
        self.record.value = value
        self.record.status = self.solver.status

    def diagonalize(self):
        self.psd_eigvals = []
        self.psd_eigvecs = []
        for expr in self.solver.psd_exprs:
            gram = expr.value
            eigvals, eigvecs = np.linalg.eigh(gram)
            self.psd_eigvals.append(eigvals)
            self.psd_eigvecs.append(eigvecs)

    def proposed_prune(self):
        if not self.psd_eigvals or not self.psd_eigvecs:
            raise ValueError('diagonalize psd blocks first')

        self.leverage = np.zeros(len(self.basis_reprs))
        null_eigvals = []

        for block_reprs, eigvals, eigvecs in zip(
            self.compiler.block_reprs,
            self.psd_eigvals,
            self.psd_eigvecs,
        ):
            null_mask = (0 <= eigvals) & (eigvals <= self.null_tol)
            if np.count_nonzero(null_mask) == 0:
                continue

            r'''
                leverage of operator O_a in the momentum-block nullspace {v_\alpha(k)}

                    l_a(k) = |N_null|^{-1} \sum_\alpha |(v_{k,\alpha})_a|^2

                normalized by inverse of total nullspace dimension |N_null| across blocks
                such that \sum_k \sum_{a \in B_k} l_a(k) = 1.
            '''
            block_leverage = np.sum(np.abs(eigvecs[:, null_mask])**2, axis=1)
            null_eigvals.extend(eigvals[null_mask])
            for rep, score in zip(block_reprs, block_leverage):
                idx = self.basis_indices[rep.canon]
                self.leverage[idx] += float(score)

        self.null_count = len(null_eigvals)
        if self.null_count == 0:
            self.to_drop = []
            self.max_null_eigval = None
            self.record.null_count = self.null_count
            self.record.max_null_eigval = self.max_null_eigval
            return self.to_drop

        self.leverage /= self.null_count
        self.max_null_eigval = float(np.max(null_eigvals)) if null_eigvals else None

        candidates = [
            (score, rep.canon)
            for rep, score in zip(self.basis_reprs, self.leverage)
            if rep.canon not in self.required_keys
            and score < self.max_drop_leverage
        ]
        # sorted by leverages in nullspace (ascending)
        candidates.sort()
        self.to_drop = [key for _, key in candidates[:self.max_drop_per_step]]

        self.record.null_count = self.null_count
        self.record.max_null_eigval = self.max_null_eigval
        return self.to_drop

    def proposed_grow(self):
        if not self.psd_eigvals or not self.psd_eigvecs:
            raise ValueError('diagonalize psd blocks first')

        known_keys = set(self.basis_indices)
        target_growth = len(self.to_drop) + self.max_net_growth_per_step
        self.to_grow = []

        # sort nullspace eigvals across all blocks
        null_dirs = []
        for block_idx, eigvals in enumerate(self.psd_eigvals):
            # momentum-block nullspace indices (ascending)
            null_indices = np.flatnonzero((0 <= eigvals) & (eigvals <= self.null_tol))
            for eig_idx in null_indices:
                null_dirs.append((eigvals[eig_idx], block_idx, eig_idx))
        null_dirs.sort()

        for _, block_idx, eig_idx in null_dirs:
            # build the null operator
            block_reprs = self.compiler.block_reprs[block_idx]
            coeffs = self.psd_eigvecs[block_idx][:, eig_idx]
            op = type(self.compiler.hamil_op)()
            for coeff, rep in zip(coeffs, block_reprs):
                if abs(coeff) < 1e-10:
                    continue
                op = op + coeff * self.compiler.fourier(rep, block_idx)

            # descendants
            desc = self.compiler.hamil_op.commutator(op)
            for pstr, coeff in desc.terms.items():
                if abs(coeff) < self.descendant_tol:
                    continue
                key = pstr.canon
                if key in known_keys:
                    continue
                known_keys.add(key)
                self.to_grow.append(pstr.canon_rep)
                if len(self.to_grow) >= target_growth:
                    break
            if len(self.to_grow) >= target_growth:
                break

        return self.to_grow

    def update(self):
        while (
            len(self.to_grow)-len(self.to_drop) < self.min_net_growth_per_step
            and self.to_drop
        ):
            # pop from the right (lowest priority)
            self.to_drop.pop()

        to_drop_set = set(self.to_drop)
        self.basis_reprs = [rep for rep in self.basis_reprs if rep.canon not in to_drop_set]
        self.basis_reprs.extend(self.to_grow)
        self.basis_indices = {rep.canon: idx for idx, rep in enumerate(self.basis_reprs)}

        self.record.to_drop = len(self.to_drop)
        self.record.to_grow = len(self.to_grow)
        self.record.net_growth = len(self.to_grow) - len(self.to_drop)
        self.history.append(self.record)
        self.record = NGARecord(basis_reps=len(self.basis_reprs))

    def step(self):
        self.build()
        summary = self.compiler.summary()
        self.solve()
        self.diagonalize()
        self.proposed_prune()
        self.proposed_grow()
        self.update()
        return summary, self.history[-1]


if __name__ == '__main__':

    from ising.ising import IsingCompiler, IsingParams, build_basis_reprs

    params = IsingParams(L=16, J=1., h=1.)
    basis_reprs = build_basis_reprs(params.L, ['I', 'X', 'ZZ'])
    required_basis_reprs = build_basis_reprs(params.L, ['I', 'X', 'ZZ'])

    runner = NGARunner(
        compiler=IsingCompiler(params),
        basis_reprs=basis_reprs,
        required_basis_reprs=required_basis_reprs,
        solver_backend='MOSEK',
        null_tol=1e-8,
        descendant_tol=1e-5,
        max_drop_leverage=5e-2,
        min_net_growth_per_step=1,
        max_net_growth_per_step=4,
        max_drop_per_step=8,
    )

    n_steps = 10
    for i in range(n_steps):
        runner.build()
        print(i, runner.compiler.summary())

        runner.solve()
        runner.diagonalize()
        runner.proposed_prune()
        runner.proposed_grow()
        runner.update()
        print(i, runner.history[-1])

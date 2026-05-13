from dataclasses import dataclass, field, fields
import math
import numpy as np

from sdp import SDPSolver

@dataclass(slots=True)
class NGAParams:
    solver_backend: str = 'MOSEK'
    solver_kwargs: dict = field(default_factory=dict)
    drop_null_tol: float = 1e-9
    grow_null_tol: float = 1e-9
    max_drop_leverage: float = 1e-2
    min_net_growth_per_step: int = 1
    max_net_growth_per_step: int = 8
    # max number of basis drop per step given by
    # max(drop_cap_base_per_step, drop_cap_rate * len(basis_reprs))
    drop_cap_base_per_step: int = 8
    drop_cap_rate: float = 0.1

    def to_dict(self):
        return {
            'solver_backend': self.solver_backend,
            'solver_kwargs': dict(self.solver_kwargs),
            'drop_null_tol': self.drop_null_tol,
            'grow_null_tol': self.grow_null_tol,
            'max_drop_leverage': self.max_drop_leverage,
            'min_net_growth_per_step': self.min_net_growth_per_step,
            'max_net_growth_per_step': self.max_net_growth_per_step,
            'drop_cap_base_per_step': self.drop_cap_base_per_step,
            'drop_cap_rate': self.drop_cap_rate,
        }

@dataclass(slots=True)
class NGARecord:
    value: float | None = None
    status: str | None = None
    basis_reps: int | None = None
    psd_dims: list[int] | None = None
    n_vars: int | None = None
    affine_rank: int | None = None
    drop_null_count: int | None = None
    grow_null_count: int | None = None
    max_drop_null_eigval: float | None = None
    max_grow_null_eigval: float | None = None
    to_drop: int | None = None
    to_grow: int | None = None
    net_growth: int | None = None
    nga_params: dict | None = None

    def to_dict(self):
        data = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if hasattr(value, 'item'):
                value = value.item()
            data[field.name] = value
        return data


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
        nga_params: NGAParams,
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
        self.required_basis_reprs = [rep.canon_rep for rep in required_basis_reprs]
        self.required_keys = {rep.canon for rep in self.required_basis_reprs}
        if self.required_keys - set(self.basis_indices):
            raise ValueError('required basis representatives must be included in basis_reprs')

        self.nga_params = nga_params

        self.solver = SDPSolver()
        self._reset_record()
        self.history = []

        self.psd_eigvals = []
        self.psd_eigvecs = []

        self.leverage = None

        # small indices have higher priority
        self.to_drop = []
        self.to_grow = []

    def _reset_record(self):
        nga_params_record = self.nga_params.to_dict()
        nga_params_record['required_basis_reprs'] = [
            str(rep.canon_rep) for rep in self.required_basis_reprs
        ]
        self.record = NGARecord(
            basis_reps = len(self.basis_reprs),
            nga_params = nga_params_record,
        )

    def build(self):
        self.compiler.compile(self.basis_reprs)
        self.solver.build(self.compiler.sdp_data())
        summary = self.compiler.summary()
        self.record.psd_dims = summary['psd_dims']
        self.record.n_vars = summary['vars']
        self.record.affine_rank = summary['affines_rank']

    def solve(self):
        value = self.solver.solve(
            backend=self.nga_params.solver_backend,
            **self.nga_params.solver_kwargs,
        )
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
            null_mask = (0 <= eigvals) & (eigvals <= self.nga_params.drop_null_tol)
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

        null_count = len(null_eigvals)
        if null_count == 0:
            self.to_drop = []
            self.record.drop_null_count = 0
            self.record.max_drop_null_eigval = None
            return self.to_drop

        self.leverage /= null_count
        max_null_eigval = float(np.max(null_eigvals)) if null_eigvals else None

        candidates = [
            (score, rep.canon)
            for rep, score in zip(self.basis_reprs, self.leverage)
            if rep.canon not in self.required_keys
            and score < self.nga_params.max_drop_leverage
        ]
        # sorted by leverages in nullspace (ascending)
        candidates.sort()
        drop_cap = max(
            self.nga_params.drop_cap_base_per_step,
            math.ceil(self.nga_params.drop_cap_rate * len(self.basis_reprs)),
        )
        self.to_drop = [key for _, key in candidates[:drop_cap]]

        self.record.drop_null_count = null_count
        self.record.max_drop_null_eigval = max_null_eigval
        return self.to_drop

    def proposed_grow(self):
        if not self.psd_eigvals or not self.psd_eigvecs:
            raise ValueError('diagonalize psd blocks first')

        basis_keys = set(self.basis_indices)
        target_growth = len(self.to_drop) + self.nga_params.max_net_growth_per_step
        self.to_grow = []

        # sort nullspace eigvals across all blocks
        null_dirs = []
        for block_idx, eigvals in enumerate(self.psd_eigvals):
            # momentum-block nullspace indices (ascending)
            null_indices = np.flatnonzero((0 <= eigvals) & (eigvals <= self.nga_params.grow_null_tol))
            for eig_idx in null_indices:
                null_dirs.append((eigvals[eig_idx], block_idx, eig_idx))
        null_dirs.sort()
        null_count = len(null_dirs)
        max_null_eigval = float(max((eigval for eigval, _, _ in null_dirs), default=0)) if null_dirs else None
        self.record.grow_null_count = null_count
        self.record.max_grow_null_eigval = max_null_eigval

        candidate_scores = {}
        candidate_reps = {}

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
                key = pstr.canon
                if key in basis_keys:
                    continue
                candidate_scores[key] = candidate_scores.get(key, 0) + float(abs(coeff))
                candidate_reps[key] = pstr.canon_rep

        candidates = sorted(
            candidate_scores,
            key=lambda key: (-candidate_scores[key], key),
        )
        self.to_grow = [candidate_reps[key] for key in candidates[:target_growth]]
        return self.to_grow

    def update(self):
        while (
            len(self.to_grow)-len(self.to_drop) < self.nga_params.min_net_growth_per_step
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
        self._reset_record()

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
        nga_params=NGAParams(
            solver_backend='MOSEK',
            drop_null_tol=1e-8,
            grow_null_tol=1e-8,
            max_drop_leverage=5e-2,
            min_net_growth_per_step=1,
            max_net_growth_per_step=8,
            drop_cap_base_per_step=8,
            drop_cap_rate=0.1,
        ),
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

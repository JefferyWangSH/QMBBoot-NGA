from dataclasses import dataclass, field, fields
import time

import numpy as np
import scipy as sp

from sdp import SDPSolver

@dataclass(slots=True)
class NGAParams:
    solver_backend: str = 'MOSEK'
    solver_kwargs: dict = field(default_factory=dict)
    drop_null_tol: float = 1e-9
    grow_null_tol: float = 1e-9
    max_drop_leverage: float = 1e-2

    def to_dict(self):
        return {
            'solver_backend': self.solver_backend,
            'solver_kwargs': dict(self.solver_kwargs),
            'drop_null_tol': self.drop_null_tol,
            'grow_null_tol': self.grow_null_tol,
            'max_drop_leverage': self.max_drop_leverage,
        }

@dataclass(slots=True)
class NGARecord:
    value: float | None = None
    objective_sense: str | None = None
    observables: dict | None = None
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
    required_basis: list[str] | None = None
    time: dict = field(default_factory=lambda: {
        'compile_time': None,
        'build_time': None,
        'solve_time': None,
    })
    nga_params: dict | None = None
    scheduler: dict | None = None

    @classmethod
    def _serialize(cls, value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, list):
            return [cls._serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._serialize(item) for key, item in value.items()}
        return value

    def to_dict(self):
        data = {}
        for field in fields(self):
            data[field.name] = self._serialize(getattr(self, field.name))
        return data


class NGARunner:
    '''
        Nullspace-guided adaptive (NGA) runner

        The model compiler must provide:

            Compiler.trans_canon
            Compiler.trans_canon_rep
            Compiler.block_reprs: list[list[BasisRep]]
            Compiler.block_momenta: list[MomentumIndex]
            Compiler.compile(basis_reprs: list[BasisRep]) -> None
            Compiler.sdp_data() -> SDPData
            Compiler.summary() -> dict
            Compiler.descendants(rep: BasisRep) -> list[tuple[BasisRep, Shift, complex]]
            Compiler.fourier_phase(q: MomentumIndex, shift: Shift) -> complex
            Compiler.nonzero_fourier(rep: BasisRep, q: MomentumIndex) -> bool
    '''
    def __init__(
        self,
        compiler,
        basis_reprs,
        required_basis_reprs,
        scheduler,
        nga_params: NGAParams,
        drop_counts: dict | None = None,
    ):
        self.compiler = compiler

        # canonical basis reprs (duplication removed)
        self.basis_reprs = []
        seen_reprs = set()
        for rep in basis_reprs:
            rep = self._canon_rep(rep)
            key = self._canon(rep)
            if key in seen_reprs:
                continue
            seen_reprs.add(key)
            self.basis_reprs.append(rep)

        self.basis_indices = {self._canon(rep): idx for idx, rep in enumerate(self.basis_reprs)}
        self.required_basis_reprs = [self._canon_rep(rep) for rep in required_basis_reprs]
        self.required_basis_keys = {self._canon(rep) for rep in self.required_basis_reprs}
        if self.required_basis_keys - set(self.basis_indices):
            raise ValueError('required basis representatives must be included in basis_reprs')

        self.scheduler = scheduler
        self.nga_params = nga_params

        self.solver = SDPSolver()
        self._reset_record()
        self.history = []

        self.psd_eigvals = []
        self.psd_eigvecs = []

        # lower-index candidates have higher priority
        self.to_drop = []
        self.to_grow = []
        self.drop_counts = {} if drop_counts is None else drop_counts

    def _reset_record(self):
        self.record = NGARecord(
            basis_reps = len(self.basis_reprs),
            nga_params = self.nga_params.to_dict(),
            required_basis = [str(rep) for rep in self.required_basis_reprs],
        )

    def _update_scheduler(self):
        self.scheduler.update(self)
        self.record.scheduler = self.scheduler.to_dict()

    def _canon(self, basis_rep):
        return self.compiler.trans_canon(basis_rep)

    def _canon_rep(self, basis_rep):
        return self.compiler.trans_canon_rep(basis_rep)

    def build(self):
        start = time.perf_counter()
        self.compiler.compile(self.basis_reprs)
        self.record.time['compile_time'] = time.perf_counter() - start

        start = time.perf_counter()
        self.solver.build(self.compiler.sdp_data())
        self.record.time['build_time'] = time.perf_counter() - start

        summary = self.compiler.summary()
        self.record.psd_dims = summary['psd_dims']
        self.record.n_vars = summary['vars']
        self.record.affine_rank = summary['affines_rank']

    def solve(self):
        start = time.perf_counter()
        self.solver.solve(
            backend=self.nga_params.solver_backend,
            **self.nga_params.solver_kwargs,
        )
        self.record.time['solve_time'] = time.perf_counter() - start

        summary = self.solver.summary()
        self.record.value = summary['value']
        self.record.objective_sense = summary['objective_sense']
        self.record.observables = summary['observables']
        self.record.status = summary['status']

    def diagonalize(self):
        self.psd_eigvals = []
        self.psd_eigvecs = []
        for expr in self.solver.psd_exprs:
            gram = expr.value
            eigvals, eigvecs = np.linalg.eigh(gram)
            self.psd_eigvals.append(eigvals)
            self.psd_eigvecs.append(eigvecs)

    def propose_pruning(self):
        if not self.psd_eigvals or not self.psd_eigvecs:
            raise ValueError('diagonalize psd blocks first')
        self._update_scheduler()

        r'''
            A null vector v_{k,\alpha} of M(k) defines

                P_{k,\alpha}
                    = L^{-1/2} \sum_a \sum_r [v_{k,\alpha}]_a e^{-i k r} T_r O_a
                    = \sum_{a,r} [v'_{k,\alpha}]_{a,r} T_r O_a,

            with [v'_{k,\alpha}]_{a,r} = L^{-1/2} [v_{k,\alpha}]_a e^{-i k r} the real-space null vector.
            \alpha enumerates the null vectors of M(k).
            The leverage of O_a in the moment nullspace is

                l_a = |N_null|^{-1} \sum_{k,\alpha} \sum_r |[v'_{k,\alpha}]_{a,r}|^2,

            so that \sum_a l_a = 1. In terms of momentum-space null vectors,
            l_a is equivalently computed as

                l_a = |N_null|^{-1} \sum_{k,\alpha} |[v_{k,\alpha}]_a|^2,

            which is implemented as below.
        '''

        leverage = np.zeros(len(self.basis_reprs))
        null_eigvals = []

        for block_reprs, eigvals, eigvecs in zip(
            self.compiler.block_reprs,
            self.psd_eigvals,
            self.psd_eigvecs,
        ):
            null_mask = np.abs(eigvals) <= self.nga_params.drop_null_tol
            if np.count_nonzero(null_mask) == 0:
                continue

            block_leverage = np.sum(np.abs(eigvecs[:, null_mask])**2, axis=1)
            null_eigvals.extend(eigvals[null_mask])
            for rep, score in zip(block_reprs, block_leverage):
                idx = self.basis_indices[self._canon(rep)]
                leverage[idx] += float(score)

        null_count = len(null_eigvals)
        if null_count == 0:
            self.to_drop = []
            self.record.drop_null_count = 0
            self.record.max_drop_null_eigval = None
            return self.to_drop

        leverage /= null_count
        max_null_eigval = float(np.max(null_eigvals)) if null_eigvals else None

        cands = [
            (score, self._canon(rep))
            for rep, score in zip(self.basis_reprs, leverage)
            if self._canon(rep) not in self.required_basis_keys
            and score < self.nga_params.max_drop_leverage
        ]
        # sort by increasing nullspace leverage
        cands.sort()
        self.to_drop = [key for _, key in cands[:self.scheduler.drop_cap]]

        self.record.drop_null_count = null_count
        self.record.max_drop_null_eigval = max_null_eigval
        return self.to_drop

    def propose_growth(self):
        if not self.psd_eigvals or not self.psd_eigvecs:
            raise ValueError('diagonalize psd blocks first')

        target_growth = len(self.to_drop) + self.scheduler.net_growth_cap

        r'''
            self.compiler.descendants(rep) calculates

                [H, O_a] = \sum_{b,s} C_{ab}(s) T_s O'_b.

            Therefore, in terms of real-space null vectors,

                (1-\Pi)[H, P_{k,\alpha}]
                    = \sum_{a,b,r,s} [v'_{k,\alpha}]_{a,r} C_{ab}(s) T_{r+s} O'_b
                    = \sum_{a,b,r,s} [v'_{k,\alpha}]_{a,r} C_{ab}(s-r) T_s O'_b
                    = \sum_{b,s} w'_{k,\alpha,b,s} T_s O'_b

            with w'_{k,\alpha,b,s} = \sum_{a,r} [v'_{k,\alpha}]_{a,r} C_{ab}(s-r).
            O'_b now enumerates the descendants that are missing in the current operator basis.
            By definition, the growth score is

                s_b = |N_null|^{-1} \sum_{k,\alpha,s} |w'_{k,\alpha,b,s}|^2.

            The code works with the momentum-space null vectors [v_{k,\alpha}]_a so that

                (1-\Pi)[H, P_{k,\alpha}]
                    = \sum_{a,b,r,s} [v'_{k,\alpha}]_{a,r} C_{ab}(s) T_{r+s} O'_b
                    = \sum_{a,b,s} [v_{k,\alpha}]_a e^{i k s} C_{ab}(s) O'_b(k).

            Writing [D^k]_{ba} = \sum_s e^{i k s} C_{ab}(s) and [v^k]_{a,\alpha} = [v_{k,\alpha}]_a gives

                (1-\Pi)[H, P_{k,\alpha}]
                    = \sum_{a,b} [D^k]_{ba} [v^k]_{a,\alpha} O'_b(k)
                    = \sum_b [D^k @ v^k]_{b,\alpha} O'_b(k).

            where we define

                w_{k,\alpha,b}
                    = [D^k @ v^k]_{b,\alpha}
                    = \sum_{a,s} [v_{k,\alpha}]_a e^{i k s} C_{ab}(s)

            and the growth score

                s_b = |N_null|^{-1} \sum_{k,\alpha} |w_{k,\alpha,b}|^2.

            The two scores are identical by noting that

                w'_{k,\alpha,b,s}
                    = \sum_{a,r} L^{-1/2} [v_{k,\alpha}]_a e^{-i k r} C_{ab}(s-r)
                    = \sum_{a,r} L^{-1/2} [v_{k,\alpha}]_a e^{-i k (s-r)} C_{ab}(r)
                    = L^{-1/2} e^{-i k s} w_{k,\alpha,b},

            and \sum_s |w'_{k,\alpha,b,s}|^2 = |w_{k,\alpha,b}|^2.
        '''

        cand_scores = {} # candidate key to its growth score
        cand_reps = {}   # candidate key to its operator representative
        null_eigvals = []

        for q, block_reprs, eigvals, eigvecs in zip(
            self.compiler.block_momenta,
            self.compiler.block_reprs,
            self.psd_eigvals,
            self.psd_eigvecs,
        ):
            null_mask = np.abs(eigvals) <= self.nga_params.grow_null_tol
            if np.count_nonzero(null_mask) == 0:
                continue

            null_eigvals.extend(eigvals[null_mask])
            null_eigvecs = eigvecs[:, null_mask]

            r'''
                sparse descendant matrix [D^k]_{ba} = \sum_s e^{i k s} C_{ab}(s)
                and self.compiler.descendants(rep) returns

                    [H, O_a(0)] = \sum_{b,s} C_{ab}(s) T_s O'(0)_b

                as a list [(O'(0)_b, s, C_{ab}(s)), ...].
            '''
            desc_keys = [] # descendants keys
            desc_rows = {} # descendant key to its row index in D^k
            d_rows = []
            d_cols = []
            d_data = []

            for a, op_a in enumerate(block_reprs):
                for op_b, s, coeff in self.compiler.descendants(op_a):
                    key = self._canon(op_b)
                    if key in self.basis_indices:
                        continue
                    if not self.compiler.nonzero_fourier(op_b, q):
                        continue

                    b = desc_rows.get(key)
                    if b is None:
                        b = len(desc_keys)
                        desc_rows[key] = b
                        desc_keys.append(key)
                        if key not in cand_reps:
                            cand_reps[key] = op_b

                    d_rows.append(b)
                    d_cols.append(a)
                    d_data.append(coeff * self.compiler.fourier_phase(q, s))

            if not d_data:
                continue

            D = sp.sparse.coo_matrix(
                (d_data, (d_rows, d_cols)),
                shape=(len(desc_keys), len(block_reprs)),
            ).tocsr() # implicitly sum over s
            w = D @ null_eigvecs
            block_scores = np.sum(np.abs(w)**2, axis=1)

            for key, score in zip(desc_keys, block_scores):
                cand_scores[key] = cand_scores.get(key, 0) + float(score)

        null_count = len(null_eigvals)
        if null_count == 0:
            self.to_grow = []
            self.record.grow_null_count = 0
            self.record.max_grow_null_eigval = None
            return self.to_grow

        self.record.grow_null_count = null_count
        self.record.max_grow_null_eigval = float(np.max(null_eigvals))
        for key in cand_scores:
            cand_scores[key] /= null_count

        if self.scheduler.reentry_penalty > 0:
            for key, count in self.drop_counts.items():
                if key in cand_scores:
                    cand_scores[key] *= (1 - self.scheduler.reentry_penalty) ** count

        cands = sorted(cand_scores, key=lambda key: (-cand_scores[key], key))
        self.to_grow = [cand_reps[key] for key in cands[:target_growth]]
        return self.to_grow

    def update(self):
        while (
            len(self.to_grow) - len(self.to_drop) < self.scheduler.net_growth_min
            and self.to_drop
        ):
            # pop those with lowest priority from the end
            self.to_drop.pop()

        to_drop = set(self.to_drop)
        for key in to_drop:
            self.drop_counts[key] = self.drop_counts.get(key, 0) + 1
        self.basis_reprs = [rep for rep in self.basis_reprs if self._canon(rep) not in to_drop]
        self.basis_reprs.extend(self.to_grow)
        self.basis_indices = {self._canon(rep): idx for idx, rep in enumerate(self.basis_reprs)}

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
        self.propose_pruning()
        self.propose_growth()
        self.update()
        return summary, self.history[-1]


if __name__ == '__main__':

    from compiler.ising import IsingCompiler, IsingParams, build_basis_reprs
    from nga_scheduler import BaseScheduler

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
        ),
        scheduler=BaseScheduler(
            net_growth_min=1,
            net_growth_cap=4,
            drop_cap=4,
            reentry_penalty=.5,
        ),
    )

    n_steps = 10
    for i in range(n_steps):
        runner.build()
        print(i, runner.compiler.summary())

        runner.solve()
        runner.diagonalize()
        runner.propose_pruning()
        runner.propose_growth()
        runner.update()
        print(i, runner.history[-1])

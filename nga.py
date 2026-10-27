from dataclasses import asdict, dataclass, field, fields
import time

import numpy as np
import scipy as sp

@dataclass(slots=True)
class NGAParams:
    solver_backend: str = 'MOSEK'
    solver_kwargs: dict = field(default_factory=dict)
    max_drop_leverage: float = 1e-2
    warm_start: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass(slots=True)
class NGARecord:
    value: float | None = None
    objective_sense: str | None = None
    observables: dict | None = None
    status: str | None = None
    n_iters: int | None = None
    warm_start: bool | None = None
    basis_reps: int | None = None
    psd_dims: list[int] | None = None
    n_vars: int | None = None
    affine_rank: int | None = None
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
        warm_data: 'SDPWarmData | None' = None,
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

        if nga_params.warm_start and nga_params.solver_backend != 'SCS_QMBBOOT':
            raise ValueError(f'warm_start is not supported by solver backend {nga_params.solver_backend}')
        if nga_params.solver_backend == 'SCS_QMBBOOT':
            from sdp import ScsSDPSolver
            self.solver = ScsSDPSolver()
        else:
            from sdp import SDPSolver
            self.solver = SDPSolver()
        self.warm_data = warm_data # export_warm() of the previous solution
        self._reset_record()
        self.history = []

        # scaled PSD duals such that \sum_b tr Y(b) = 1
        self.psd_duals = []

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
            warm_start=self.warm_data,
            **self.nga_params.solver_kwargs,
        )
        self.record.time['solve_time'] = time.perf_counter() - start

        summary = self.solver.summary()
        self.record.value = summary['value']
        self.record.objective_sense = summary['objective_sense']
        self.record.observables = summary['observables']
        self.record.status = summary['status']
        self.record.n_iters = summary['num_iters']
        self.record.warm_start = self.warm_data is not None

        # keep the last solution as the warm data of the next step
        if self.nga_params.warm_start and self.solver.status in ('optimal', 'optimal_inaccurate'):
            self.warm_data = self.solver.export_warm()

    def load_psd_duals(self):
        self.psd_duals = []
        if self.record.status is None:
            raise ValueError('solve the SDP first')
        psd_duals = self.solver.psd_duals()
        psd_trace = sum(np.trace(Y).real for Y in psd_duals)
        if not psd_trace > 0:
            raise ValueError(f'PSD duals have a non-positive total trace {psd_trace:.3e}')
        self.psd_duals = [Y / psd_trace for Y in psd_duals]

    def propose_pruning(self):
        if not self.psd_duals:
            raise ValueError('load PSD duals first')
        self._update_scheduler()

        r'''
            b = (k, p) labels a PSD block M(b) of the SDP, with momentum k and parities p.
            The dual-weighted leverage of O_a is

                \ell_a = \sum_b [Y(b)]_{aa},

            where the PSD duals Y(b) are scaled such that \sum_b tr Y(b) = 1.
        '''
        leverage = np.zeros(len(self.basis_reprs))
        for block_reprs, Y in zip(self.compiler.block_reprs, self.psd_duals, strict=True):
            indices = [self.basis_indices[self._canon(rep)] for rep in block_reprs]
            np.add.at(leverage, indices, Y.diagonal().real)

        cands = [
            (score, self._canon(rep))
            for rep, score in zip(self.basis_reprs, leverage)
            if self._canon(rep) not in self.required_basis_keys
            and score < self.nga_params.max_drop_leverage
        ]
        # sort by increasing leverage
        cands.sort()
        self.to_drop = [key for _, key in cands[:self.scheduler.drop_cap]]
        return self.to_drop

    def propose_growth(self):
        if not self.psd_duals:
            raise ValueError('load PSD duals first')

        target_growth = len(self.to_drop) + self.scheduler.net_growth_cap

        r'''
            The dual-weighted growth score of a descendant \tilde O_\mu missing in the current basis is

                s_\mu = \sum_b [D(b) Y(b) D(b)^\dag]_{\mu\mu},

            where D_{\mu a}(b) = \sum_s e^{i k s} C_{a\mu}(s) for b = (k, p),
            and C_{a\mu}(s) are the expansion coefficients of the Hamiltonian dynamics of O_a(0)
            with the current operator space projected out,

                (1 - \Pi)[H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) \tilde O_\mu(s).
        '''
        cand_scores = {} # candidate key to its growth score
        cand_reps = {}   # candidate key to its operator representative

        for q, block_reprs, Y in zip(
            self.compiler.block_momenta,
            self.compiler.block_reprs,
            self.psd_duals,
            strict=True,
        ):
            # self.compiler.descendants(O_a) returns all terms of [H, O_a(0)]
            # as a list [(O_\mu(0), s, coeff), ...].
            desc_keys = [] # descendants keys
            desc_rows = {} # descendant key to its row index in D(b)
            d_rows = []
            d_cols = []
            d_data = []

            for a, op_a in enumerate(block_reprs):
                for op_mu, s, coeff in self.compiler.descendants(op_a):
                    key = self._canon(op_mu)
                    if key in self.basis_indices:
                        continue
                    if not self.compiler.nonzero_fourier(op_mu, q):
                        continue

                    mu = desc_rows.get(key)
                    if mu is None:
                        mu = len(desc_keys)
                        desc_rows[key] = mu
                        desc_keys.append(key)
                        if key not in cand_reps:
                            cand_reps[key] = op_mu

                    d_rows.append(mu)
                    d_cols.append(a)
                    d_data.append(coeff * self.compiler.fourier_phase(q, s))

            if not d_data:
                continue

            D = sp.sparse.coo_matrix(
                (d_data, (d_rows, d_cols)),
                shape=(len(desc_keys), len(block_reprs)),
            ).tocsr() # implicitly sum over s

            # diag(D(b) Y(b) D^dag(b))
            block_scores = np.asarray(D.conj().multiply(D @ Y).sum(axis=1)).ravel().real

            for key, score in zip(desc_keys, block_scores):
                cand_scores[key] = cand_scores.get(key, 0) + float(score)

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
        self.load_psd_duals()
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
        runner.load_psd_duals()
        runner.propose_pruning()
        runner.propose_growth()
        runner.update()
        print(i, runner.history[-1])

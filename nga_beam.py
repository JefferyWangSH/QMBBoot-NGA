from dataclasses import dataclass, field, fields
import time

import numpy as np
import scipy as sp

from nga import NGAParams
from sdp import SDPSolver

@dataclass(slots=True)
class NGABeamRecord:
    selected: int | None = None
    value: float | None = None
    objective_sense: str | None = None
    observables: dict | None = None
    status: str | None = None
    basis_reps: int | None = None
    psd_dims: list[int] | None = None
    n_vars: int | None = None
    affine_rank: int | None = None
    to_drop: int | None = None
    to_grow: int | None = None
    net_growth: int | None = None
    scheduler: dict | None = None
    nga_params: dict | None = None
    candidates: list[dict] = field(default_factory=list)

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

@dataclass(slots=True)
class NGAEvalRecord:
    value: float | None = None
    objective_sense: str | None = None
    observables: dict | None = None
    status: str | None = None
    basis_reps: int | None = None
    psd_dims: list[int] | None = None
    n_vars: int | None = None
    affine_rank: int | None = None
    required_basis: list[str] | None = None
    time: dict = field(default_factory=lambda: {
        'compile_time': None,
        'build_time': None,
        'solve_time': None,
    })
    nga_params: dict | None = None
    drop_null_count: int | None = None
    grow_null_count: int | None = None
    max_drop_null_eigval: float | None = None
    max_grow_null_eigval: float | None = None

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

@dataclass(slots=True)
class NGABeamCandidate:
    record: NGAEvalRecord
    basis: list
    move: dict = field(default_factory=lambda: {'drop': [], 'grow': []})
    probs: dict = field(default_factory=lambda: {'drop': [], 'grow': []})
    grow_scores: list = field(default_factory=list)

    def to_dict(self):
        return {
            'record': self.record.to_dict(),
            'basis': len(self.basis),
            'move': {
                'drop': [str(rep) for rep in self.move['drop']],
                'grow': [str(rep) for rep in self.move['grow']],
            },
            'probs': self.probs,
        }


class NGABeamRunner:
    def __init__(
        self,
        compiler,
        basis_reprs,
        required_basis_reprs,
        scheduler,
        nga_params: NGAParams,
        drop_counts: dict | None = None,
        max_workers: int = 1,
    ):
        self.compiler = compiler

        self.basis_reprs = []
        reprs_seen = set()
        for rep in basis_reprs:
            rep = self._canon_rep(rep)
            key = self._canon(rep)
            if key in reprs_seen:
                continue
            reprs_seen.add(key)
            self.basis_reprs.append(rep)

        self.required_basis_reprs = [self._canon_rep(rep) for rep in required_basis_reprs]
        self.required_basis_keys = {self._canon(rep) for rep in required_basis_reprs}
        
        self.scheduler = scheduler
        self.nga_params = nga_params
        self.max_workers = max_workers
        self.solver = SDPSolver()

        self.drop_counts = {} if drop_counts is None else drop_counts
        self.to_drop = []
        self.to_grow = []
        self.psd_eigvals = []
        self.psd_eigvecs = []
        self.cands = []
        self.beam_record = NGABeamRecord(
            nga_params=self.nga_params.to_dict(),
        )
        self.history = []

    def _canon(self, basis_rep):
        return self.compiler.trans_canon(basis_rep)

    def _canon_rep(self, basis_rep):
        return self.compiler.trans_canon_rep(basis_rep)

    def evaluate(self, basis_cand):
        eval_record = NGAEvalRecord(
            basis_reps=len(basis_cand),
            nga_params=self.nga_params.to_dict(),
            required_basis=[str(rep) for rep in self.required_basis_reprs],
        )

        start = time.perf_counter()
        self.compiler.compile(basis_cand)
        eval_record.time = {'compile_time': time.perf_counter() - start}

        start = time.perf_counter()
        self.solver.build(self.compiler.sdp_data())
        eval_record.time['build_time'] = time.perf_counter() - start

        summary = self.compiler.summary()
        eval_record.psd_dims = summary['psd_dims']
        eval_record.n_vars = summary['vars']
        eval_record.affine_rank = summary['affines_rank']

        start = time.perf_counter()
        self.solver.solve(
            backend=self.nga_params.solver_backend,
            **self.nga_params.solver_kwargs,
        )
        eval_record.time['solve_time'] = time.perf_counter() - start

        summary = self.solver.summary()
        eval_record.value = summary['value']
        eval_record.objective_sense = summary['objective_sense']
        eval_record.observables = summary['observables']
        eval_record.status = summary['status']

        self.psd_eigvals = []
        self.psd_eigvecs = []
        for expr in self.solver.psd_exprs:
            gram = expr.value
            eigvals, eigvecs = np.linalg.eigh(gram)
            self.psd_eigvals.append(eigvals)
            self.psd_eigvecs.append(eigvecs)
        return eval_record

    def _drop_scores(self, basis_cand, eval_record: NGAEvalRecord):
        leverage = np.zeros(len(basis_cand))
        null_eigvals = []

        basis_indices = {
            self._canon(rep): idx
            for idx, rep in enumerate(basis_cand)
        }

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
                idx = basis_indices[self._canon(rep)]
                leverage[idx] += float(score)

        null_count = len(null_eigvals)
        if null_count == 0:
            eval_record.drop_null_count = 0
            eval_record.max_drop_null_eigval = None
            return []
        leverage /= null_count

        scores = []
        for rep, score in zip(basis_cand, leverage):
            key = self._canon(rep)
            if key in self.required_basis_keys:
                continue
            if score >= self.nga_params.max_drop_leverage:
                continue
            scores.append((float(score), key, rep))
        scores.sort(key=lambda item: (item[0], item[1]))

        eval_record.drop_null_count = null_count
        eval_record.max_drop_null_eigval = float(np.max(null_eigvals))
        return scores

    def _grow_scores(self, basis_cand, eval_record: NGAEvalRecord):
        basis_keys = {self._canon(rep) for rep in basis_cand}
        cand_scores = {}
        cand_reps = {}
        null_eigvals = []

        for n, block_reprs, eigvals, eigvecs in zip(
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
            k = 2 * np.pi * n / self.compiler.L

            desc_rows = {}
            desc_keys = []
            rows = []
            cols = []
            data = []

            for a_idx, rep in enumerate(block_reprs):
                for desc_rep, s, coeff in self.compiler.descendants(rep):
                    desc_key = self._canon(desc_rep)
                    if desc_key in basis_keys:
                        continue
                    if not self.compiler.nonzero_fourier(desc_rep, n):
                        continue

                    row = desc_rows.get(desc_key)
                    if row is None:
                        row = len(desc_keys)
                        desc_rows[desc_key] = row
                        desc_keys.append(desc_key)
                        if desc_key not in cand_reps:
                            cand_reps[desc_key] = desc_rep

                    rows.append(row)
                    cols.append(a_idx)
                    data.append(coeff * np.exp(1j * k * s))

            if not data:
                continue

            D = sp.sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(len(desc_keys), len(block_reprs)),
            ).tocsr()
            w = D @ null_eigvecs
            block_scores = np.sum(np.abs(w)**2, axis=1)

            for desc_key, score in zip(desc_keys, block_scores):
                cand_scores[desc_key] = cand_scores.get(desc_key, 0) + float(score)

        if self.scheduler.reentry_penalty > 0:
            for key, count in self.drop_counts.items():
                if key in cand_scores:
                    cand_scores[key] *= (1 - self.scheduler.reentry_penalty) ** count

        scores = [(float(score), key, cand_reps[key]) for key, score in cand_scores.items()]
        scores.sort(key=lambda item: (-item[0], item[1]))

        eval_record.grow_null_count = len(null_eigvals)
        eval_record.max_grow_null_eigval = float(np.max(null_eigvals)) if null_eigvals else None
        return scores

    def _grow(self, grow_scores):
        return [rep for _, _, rep in grow_scores[:self.scheduler.growth_cap]]

    def _replace(self, drop_scores, grow_scores):
        n_drop = min(self.scheduler.replace_cap, len(drop_scores))
        n_grow = min(self.scheduler.replace_cap, len(grow_scores))
        n_replace = min(n_drop, n_grow)
        if n_replace == 0:
            return (
                {'drop': [], 'grow': []},
                {'drop': [], 'grow': []},
            )

        if self.scheduler.drop_temperature > 0:
            scores = np.array([score for score, _, _ in drop_scores], dtype=float)
            logits = -scores / self.scheduler.drop_temperature
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            drop_indices = np.random.choice(
                len(drop_scores),
                size=n_replace,
                replace=False,
                p=probs,
            )
            to_drop = [drop_scores[i][2] for i in drop_indices]
            drop_probs = probs[drop_indices].tolist()
        else:
            to_drop = [rep for _, _, rep in drop_scores[:n_replace]]
            drop_probs = [1.] * n_replace

        if self.scheduler.grow_temperature > 0:
            scores = np.array([score for score, _, _ in grow_scores], dtype=float)
            logits = scores / self.scheduler.grow_temperature
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            grow_indices = np.random.choice(
                len(grow_scores),
                size=n_replace,
                replace=False,
                p=probs,
            )
            to_grow = [grow_scores[i][2] for i in grow_indices]
            grow_probs = probs[grow_indices].tolist()
        else:
            to_grow = [rep for _, _, rep in grow_scores[:n_replace]]
            grow_probs = [1.] * n_replace

        return (
            {'drop': to_drop, 'grow': to_grow},
            {'drop': drop_probs, 'grow': grow_probs},
        )

    def _apply_move(self, basis_reprs, move):
        basis_map = {
            self._canon(rep): self._canon_rep(rep)
            for rep in basis_reprs
        }

        for rep in move.get('drop', []):
            key = self._canon(rep)
            if key in basis_map:
                basis_map.pop(key)
            else:
                raise KeyError(key)

        for rep in move.get('grow', []):
            rep = self._canon_rep(rep)
            basis_map[self._canon(rep)] = rep

        return list(basis_map.values())

    def step(self):
        self.beam_record = NGABeamRecord(
            nga_params=self.nga_params.to_dict(),
        )

        self.scheduler.update(self)
        self.beam_record.scheduler = self.scheduler.to_dict()

        root_basis = list(self.basis_reprs)
        root_eval_record = self.evaluate(root_basis)
        root_drop_scores = self._drop_scores(root_basis, root_eval_record)
        root_grow_scores = self._grow_scores(root_basis, root_eval_record)

        root = NGABeamCandidate(
            record=root_eval_record,
            basis=root_basis,
            move={'drop': [], 'grow': []},
            grow_scores=root_grow_scores,
        )
        self.cands = [root]

        jobs = []
        for _ in range(self.scheduler.replace_num):
            move, probs = self._replace(root_drop_scores, root_grow_scores)
            basis_cand = self._apply_move(root_basis, move)
            jobs.append((basis_cand, move, probs))

        def eval_job(job):
            basis_cand, move, probs = job
            runner = NGABeamRunner(
                compiler=self.compiler.clone(),
                basis_reprs=basis_cand,
                required_basis_reprs=self.required_basis_reprs,
                scheduler=self.scheduler,
                nga_params=self.nga_params,
                drop_counts=dict(self.drop_counts),
            )
            eval_record = runner.evaluate(basis_cand)
            grow_scores = runner._grow_scores(basis_cand, eval_record)
            return NGABeamCandidate(
                record=eval_record,
                basis=basis_cand,
                move=move,
                probs=probs,
                grow_scores=grow_scores,
            )

        max_workers = min(len(jobs), self.max_workers)
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                self.cands.extend(executor.map(eval_job, jobs))
        else:
            self.cands.extend(eval_job(job) for job in jobs)

        if self.compiler.obj_sense == 'min':
            selected = max(range(len(self.cands)), key=lambda i: self.cands[i].record.value)
        else:
            selected = min(range(len(self.cands)), key=lambda i: self.cands[i].record.value)
        best = self.cands[selected]
        grow_move = {'drop': [], 'grow': self._grow(best.grow_scores)}

        if best.move['drop']:
            # replaced basis            
            for rep in best.move['drop']:
                key = self._canon(rep)
                self.drop_counts[key] = self.drop_counts.get(key, 0) + 1

        self.beam_record.basis_reps = len(best.basis)
        self.beam_record.psd_dims = best.record.psd_dims
        self.beam_record.n_vars = best.record.n_vars
        self.beam_record.affine_rank = best.record.affine_rank

        # apply the move
        self.basis_reprs = self._apply_move(best.basis, grow_move)
        self.to_drop = list(best.move['drop'])
        self.to_grow = list(best.move['grow']) + grow_move['grow']
    
        self.beam_record.selected = selected
        self.beam_record.value = best.record.value
        self.beam_record.objective_sense = best.record.objective_sense
        self.beam_record.observables = best.record.observables
        self.beam_record.status = best.record.status
        self.beam_record.candidates = [cand.to_dict() for cand in self.cands]
        self.beam_record.to_drop = len(self.to_drop)
        self.beam_record.to_grow = len(self.to_grow)
        self.beam_record.net_growth = len(self.to_grow) - len(self.to_drop)

        self.history.append(self.beam_record)
        return self.history[-1]


if __name__ == '__main__':

    from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
    from operators.majorana import MajoranaMonomial
    from nga_scheduler import BaseBeamScheduler

    params = HubbardParams(L=10, t=1., U=4., n_particles=10)
    basis_reprs = build_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    )
    required_basis_reprs = [
        MajoranaMonomial.from_str(params.L, s).trans_canon_rep
        for s in [
            'I',
            '0u+',
            '0u-',
            '0d+',
            '0d-',
            '0u+ 0u-',
            '0d+ 0d-',
        ]
    ]

    runner = NGABeamRunner(
        compiler=HubbardCompiler(params),
        basis_reprs=basis_reprs,
        required_basis_reprs=required_basis_reprs,
        nga_params=NGAParams(
            solver_backend='MOSEK',
            drop_null_tol=1e-8,
            grow_null_tol=1e-8,
            max_drop_leverage=5e-2,
        ),
        scheduler=BaseBeamScheduler(
            growth_cap=4,
            replace_num=2,
            replace_cap=4,
            grow_temperature=0.1,
            drop_temperature=0.1,
            reentry_penalty=0.5,
        ),
        max_workers=2,
    )

    n_steps = 10
    for i in range(n_steps):
        record = runner.step()
        print(i, record.to_dict())

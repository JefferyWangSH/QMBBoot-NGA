from dataclasses import dataclass, field
import cvxpy as cp
import numpy as np
import scipy as sp
from scipy.linalg import qr

@dataclass(slots=True)
class LinearExpr:
    '''
        Compiled affine expression of moment variables
    '''
    terms: dict[int, float|complex] = field(default_factory=dict)
    const: float|complex = 0

    def to_real(self) -> list['LinearExpr']:
        real_terms = {}
        imag_terms = {}
        for idx, coeff in self.terms.items():
            if isinstance(coeff, complex):
                real, imag = coeff.real, coeff.imag
                if real != 0:
                    real_terms[idx] = real
                if imag != 0:
                    imag_terms[idx] = imag
            elif coeff != 0:
                real_terms[idx] = coeff

        exprs = []

        real_expr = LinearExpr(
            terms=real_terms,
            const=self.const.real if isinstance(self.const, complex) else self.const,
        )
        if real_expr.terms or real_expr.const != 0:
            exprs.append(real_expr)

        imag_expr = LinearExpr(
            terms=imag_terms,
            const=self.const.imag if isinstance(self.const, complex) else 0,
        )
        if imag_expr.terms or imag_expr.const != 0:
            exprs.append(imag_expr)

        return exprs


@dataclass(slots=True)
class PSDConstraints:
    r'''
        M_{ij} = C_{ij} + \sum_k A^k_{ij} m_k
    '''
    n_vars: int
    dim: int

    rows: list[list[int]] = field(init=False)
    cols: list[list[int]] = field(init=False)
    vals: list[list[float|complex]] = field(init=False)

    const_rows: list[int] = field(init=False)
    const_cols: list[int] = field(init=False)
    const_vals: list[float|complex] = field(init=False)

    def __post_init__(self):
        self.rows = [[] for _ in range(self.n_vars)]
        self.cols = [[] for _ in range(self.n_vars)]
        self.vals = [[] for _ in range(self.n_vars)]
        self.const_rows = []
        self.const_cols = []
        self.const_vals = []

    def add(self, row: int, col: int, expr: LinearExpr):
        if expr.const != 0:
            self.const_rows.append(row)
            self.const_cols.append(col)
            self.const_vals.append(expr.const)

        for idx, coeff in expr.terms.items():
            if coeff == 0:
                continue
            self.rows[idx].append(row)
            self.cols[idx].append(col)
            self.vals[idx].append(coeff)

    def matrices(self):
        shape = (self.dim, self.dim)
        const = sp.sparse.csr_matrix(
            (self.const_vals, (self.const_rows, self.const_cols)),
            shape=shape,
        )
        coeffs = [
            sp.sparse.csr_matrix((vals, (rows, cols)), shape=shape)
            for rows, cols, vals in zip(self.rows, self.cols, self.vals)
        ]
        return const, coeffs


@dataclass(slots=True)
class AffineConstraints:
    n_vars: int
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    vals: list[float|complex] = field(default_factory=list)
    n_rows: int = 0

    def add(self, expr: LinearExpr):
        for real_expr in expr.to_real():
            for idx, coeff in real_expr.terms.items():
                self.rows.append(self.n_rows)
                self.cols.append(idx)
                self.vals.append(coeff)
            if real_expr.const != 0:
                self.rows.append(self.n_rows)
                self.cols.append(self.n_vars)
                self.vals.append(real_expr.const)
            self.n_rows += 1

    def matrix(self, prune: bool = False, tol: float = 1e-10):
        mat = sp.sparse.csr_matrix(
            (self.vals, (self.rows, self.cols)),
            shape=(self.n_rows, self.n_vars+1),
        )
        if not prune or self.n_rows == 0:
            return mat, self.n_rows

        _, r, piv = qr(mat.toarray().T, pivoting=True, mode='economic')
        rank = int(np.linalg.matrix_rank(r, tol=tol))
        return mat[sorted(piv[:rank])], rank


@dataclass(slots=True)
class SDPData:
    var_cpx: bool
    n_vars: int
    objective: LinearExpr
    psd_blocks: list[PSDConstraints]
    affines_mat: sp.sparse.csr_matrix


class SDPSolver:
    def __init__(self):
        self.vars = None
        self.constraints = None
        self.problem = None
        self.objective = None
        self.status = None

    def build(self, data: SDPData):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ variables
        self.vars = cp.Variable(data.n_vars, complex=data.var_cpx, name='vars')

        self.constraints = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PSD constraints
        for block in data.psd_blocks:
            const, coeffs = block.matrices()
            psd_mat = cp.Constant(const)
            for idx, coeff in enumerate(coeffs):
                if coeff.nnz:
                    psd_mat += self.vars[idx] * coeff
            self.constraints.append(psd_mat >> 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ affine constraints
        if data.affines_mat.shape[0] > 0:
            m_aug = cp.hstack([self.vars, cp.Constant([1.])])
            self.constraints.append(data.affines_mat @ m_aug == 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ objective and problem
        obj = cp.Constant(data.objective.const)
        for idx, coeff in data.objective.terms.items():
            obj += coeff * self.vars[idx]
        if obj.is_complex():
            obj = cp.real(obj)
        self.objective = cp.Minimize(obj)

        self.problem = cp.Problem(self.objective, self.constraints)
        return self.problem

    def solve(self, solver: str = 'SCS', **kwargs):
        if self.problem is None:
            raise ValueError('problem has not been built')
        value = self.problem.solve(solver=solver, **kwargs)
        self.status = self.problem.status
        return value

    def summary(self):
        assert self.problem is not None and self.status is not None
        return {
            'n_vars': self.vars.shape[0],
            'n_constraints': len(self.constraints),
            'status': self.problem.status,
            'value': self.problem.value,
            'solver_name': self.problem.solver_stats.solver_name,
            'solve_time': self.problem.solver_stats.solve_time,
            'num_iters': self.problem.solver_stats.num_iters,
        }


if __name__ == '__main__':

    from ising.ising import IsingParams, IsingCompiler

    params=IsingParams(L=16, J=1., h=1.)
    basis0 = ['I', 'X', 'Y', 'Z', 'ZZ']
    basis1 = basis0 + ['XX', 'YY', 'XZ', 'XY', 'YX', 'YZ', 'ZX', 'ZY']
    basis2 = basis0 + ['XX', 'YY', 'XXX', 'YYY', 'ZZZ']

    compiler = IsingCompiler(params=params)
    compiler.compile(
        basis=basis1 # basis0, basis1, basis2
    )
    print(compiler.summary())

    solver = SDPSolver()
    solver.build(compiler.sdp_data())
    solver.solve(solver='SCS', eps=1e-4, max_iters=10_000)

    print(solver.problem.value/params.L)
    print(solver.problem.status)
    print(solver.summary())

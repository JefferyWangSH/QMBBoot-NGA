from dataclasses import dataclass, field
import cvxpy as cp
import numpy as np
import scipy as sp
import sksparse.spqr as spqr

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
               = C_{ij} + \sum_k A_{i,j,k} m_k

        implemented as a sparse matrix multiplication

        e.g. for momentum PSD block
        each entry M_{ij} involves ~L number of vars (through fourier transformation);
        this leads to roughly ~ (dim^2 * L) number of non-zero entries in A_{ij,k}
        and a quite sparse density ~ (L / n_vars).
    '''
    n_vars: int
    dim: int

    # flattened view of A^k_{ij} into a sparse matrix A_{ij,k}
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    vals: list[float|complex] = field(default_factory=list)

    # flattened view of C_{ij}
    const_rows: list[int] = field(default_factory=list)
    const_vals: list[float|complex] = field(default_factory=list)

    def add(self, i: int, j: int, expr: LinearExpr):
        idx = i + j * self.dim

        if expr.const != 0:
            self.const_rows.append(idx)
            self.const_vals.append(expr.const)

        for k, coeff in expr.terms.items():
            if coeff == 0:
                continue
            self.rows.append(idx)
            self.cols.append(k)
            self.vals.append(coeff)

    def matrix_expr(self, vars: cp.Variable) -> cp.Expression:
        '''
            get CVXPY expression for the PSD matrix
        '''
        shape = (self.dim*self.dim, self.n_vars)

        A = sp.sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=shape)
        dtype = np.result_type(np.asarray(self.vals), np.asarray(self.const_vals), float)
        C = np.zeros(self.dim*self.dim, dtype=dtype)
        np.add.at(C, self.const_rows, self.const_vals)

        res = A @ vars + cp.Constant(C)
        return cp.reshape(res, (self.dim, self.dim), order='F')


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

        r, piv = spqr.spqr(mat.T.tocsc(), mode='r', tol=tol)
        rank = int(np.count_nonzero(np.abs(r.diagonal()) > tol))
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
        self.psd_exprs = None
        self.constraints = None
        self.problem = None
        self.objective = None
        self.status = None

    def build(self, data: SDPData):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ variables
        self.vars = cp.Variable(data.n_vars, complex=data.var_cpx, name='vars')
        self.constraints = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PSD constraints
        self.psd_exprs = []
        for block in data.psd_blocks:
            expr = block.matrix_expr(self.vars)
            self.psd_exprs.append(expr)
            self.constraints.append(expr >> 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ affine constraints
        if data.affines_mat.shape[0] > 0:
            m_aug = cp.hstack([self.vars, cp.Constant([1.])])
            self.constraints.append(data.affines_mat @ m_aug == 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ objective
        obj = cp.Constant(data.objective.const)
        for idx, coeff in data.objective.terms.items():
            obj += coeff * self.vars[idx]
        if obj.is_complex():
            obj = cp.real(obj)
        self.objective = cp.Minimize(obj)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ problem
        self.problem = cp.Problem(self.objective, self.constraints)
        return self.problem

    def solve(self, backend: str = 'SCS', **kwargs):
        if self.problem is None:
            raise ValueError('problem has not been built')
        value = self.problem.solve(solver=backend, **kwargs)
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
    solver.solve(backend='SCS', eps=1e-4, max_iters=10_000)

    print(solver.problem.value)
    print(solver.problem.status)
    print(solver.summary())

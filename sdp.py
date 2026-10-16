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

    def affine_map(self) -> tuple[sp.sparse.csc_matrix, np.ndarray]:
        '''
            sparse A_{ij,k} of shape (dim^2, n_vars) and dense C_{ij} of shape (dim^2,),
            such that vec(M) = A @ m + C with column-major vec (idx = i + j*dim)
        '''
        shape = (self.dim*self.dim, self.n_vars)

        A = sp.sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=shape)
        dtype = np.result_type(np.asarray(self.vals), np.asarray(self.const_vals), float)
        C = np.zeros(self.dim*self.dim, dtype=dtype)
        np.add.at(C, self.const_rows, self.const_vals)
        return A, C


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

        mat_t = mat.T.tocsc()
        mat_t.indices = mat_t.indices.astype(np.int64, copy=False)
        mat_t.indptr = mat_t.indptr.astype(np.int64, copy=False)
        r, piv = spqr.spqr(mat_t, mode='r', tol=tol)
        rank = int(np.count_nonzero(np.abs(r.diagonal()) > tol))
        return mat[sorted(piv[:rank])], rank


@dataclass(slots=True)
class SDPData:
    var_cpx: bool
    n_vars: int
    objective: LinearExpr
    objective_sense: str
    psd_blocks: list[PSDConstraints]
    affines_mat: sp.sparse.csr_matrix
    observables: dict[str, LinearExpr | list[LinearExpr]]
    energy_ineqs: list[LinearExpr]


class SDPSolver:
    def __init__(self):
        self.vars = None
        self.psd_exprs = None
        self.constraints = None
        self.problem = None
        self.objective = None
        self.objective_sense = None
        self.observables = None
        self.status = None

    def _to_cp_expr(self, expr: LinearExpr):
        res = cp.Constant(expr.const)
        for idx, coeff in expr.terms.items():
            res += coeff * self.vars[idx]
        return res

    def build(self, data: SDPData):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ variables
        self.vars = cp.Variable(data.n_vars, complex=data.var_cpx, name='vars')
        self.constraints = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PSD constraints
        self.psd_exprs = []
        for block in data.psd_blocks:
            A, C = block.affine_map()
            expr = cp.reshape(A @ self.vars + cp.Constant(C), (block.dim, block.dim), order='F')
            self.psd_exprs.append(expr)
            self.constraints.append(expr >> 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ affine constraints
        if data.affines_mat.shape[0] > 0:
            m_aug = cp.hstack([self.vars, cp.Constant([1.])])
            self.constraints.append(data.affines_mat @ m_aug == 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ objective
        obj = self._to_cp_expr(data.objective)
        if obj.is_complex():
            obj = cp.real(obj)

        self.objective_sense = data.objective_sense
        if self.objective_sense == 'min':
            self.objective = cp.Minimize(obj)
        elif self.objective_sense == 'max':
            self.objective = cp.Maximize(obj)
        else:
            raise ValueError(f'unknown objective sense: {self.objective_sense}')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ energy window
        for ineq in data.energy_ineqs:
            expr = self._to_cp_expr(ineq)
            if expr.is_complex():
                expr = cp.real(expr)
            self.constraints.append(expr >= 0)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ uncertified observables
        self.observables = {}
        for name, obs in data.observables.items():
            if isinstance(obs, list):
                self.observables[name] = [self._to_cp_expr(item) for item in obs]
            else:
                self.observables[name] = self._to_cp_expr(obs)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ problem
        self.problem = cp.Problem(self.objective, self.constraints)
        return self.problem

    def solve(self, backend: str = 'SCS', **kwargs):
        if self.problem is None:
            raise ValueError('problem has not been built')
        value = self.problem.solve(solver=backend, **kwargs)
        self.status = self.problem.status
        return value

    def psd_values(self) -> list[np.ndarray]:
        '''
            PSD blocks M_k at the current solution, each a (dim, dim) matrix
        '''
        if self.status is None:
            raise ValueError('problem has not been solved')
        return [expr.value for expr in self.psd_exprs]

    def summary(self):
        assert self.problem is not None and self.status is not None
        return {
            'n_vars': self.vars.shape[0],
            'n_constraints': len(self.constraints),
            'status': self.problem.status,
            'value': self.problem.value,
            'objective_sense': self.objective_sense,
            'observables': {
                name: [item.value for item in obs] if isinstance(obs, list) else obs.value
                for name, obs in self.observables.items()
            },
            'solver_name': self.problem.solver_stats.solver_name,
            'solve_time': self.problem.solver_stats.solve_time,
            'num_iters': self.problem.solver_stats.num_iters,
        }


class ScsSDPSolver:
    '''
        SDP solver calling scs-python directly with native complex PSD cones.
        It shares the same interface as SDPSolver:
        build(data), solve(backend, **kwargs), summary(), psd_values().
    '''

    name = 'SCS_SSNAL'

    # SCS status_val -> cvxpy status (as in cvxpy's SCS interface)
    STATUS = {
        1: 'optimal',
        2: 'optimal_inaccurate',
        -1: 'unbounded',
        -6: 'unbounded_inaccurate',
        -2: 'infeasible',
        -7: 'infeasible_inaccurate',
        -3: 'solver_error',
        -4: 'solver_error',
        -5: 'solver_error',
    }
    SOLVED = ('optimal', 'optimal_inaccurate')
    INFEASIBLE = ('infeasible', 'infeasible_inaccurate')
    UNBOUNDED = ('unbounded', 'unbounded_inaccurate')

    def __init__(self):
        self.data = None        # SDPData
        self.problem = None     # (scs_data, cone, layout) from _to_scs
        self.scs_output = None  # raw scs output {'x', 'y', 's', 'info'}
        self.status = None

    def build(self, data: SDPData):
        self.data, self.problem = data, self._to_scs(data)
        self.scs_output = self.status = None
        return self.problem

    def solve(self, backend: str | None = None, **kwargs):
        '''
            `backend` is ignored (interface compatibility). kwargs are SCS settings, translated
            as cvxpy does: eps -> eps_abs = eps_rel = eps; default eps_abs = eps_rel = 1e-5.
        '''
        import scs
        if self.problem is None:
            raise ValueError('problem has not been built')
        opts = dict(kwargs)
        if 'eps' in opts:
            opts['eps_abs'] = opts['eps_rel'] = opts.pop('eps')
        opts.setdefault('eps_abs', 1e-5)
        opts.setdefault('eps_rel', 1e-5)
        opts.setdefault('verbose', False)

        scs_data, cone, layout = self.problem
        self.scs_output = scs.SCS(scs_data, cone, **opts).solve()
        info = self.scs_output['info']
        self.status = self.STATUS.get(info['status_val'], 'solver_error')
        if self.status == 'solver_error':
            raise RuntimeError(f"SCS failed: {info['status']} (status_val {info['status_val']})")
        return self._value()

    def psd_values(self) -> list[np.ndarray]:
        '''
            PSD blocks M_k = C_k + A_k x at the current solution, each a (dim, dim) matrix
        '''
        if self.status not in self.SOLVED:
            raise ValueError(f'no solution available (status: {self.status})')
        x = self.scs_output['x']
        values = []
        for block in self.data.psd_blocks:
            A, C = block.affine_map()
            values.append((A @ x + C).reshape((block.dim, block.dim), order='F'))
        return values

    def _value(self):
        # as cvxpy: +inf (min) / -inf (max) if infeasible, the opposite if unbounded
        _, _, layout = self.problem
        if self.status in self.INFEASIBLE:
            return layout['obj_sign'] * np.inf
        if self.status in self.UNBOUNDED:
            return -layout['obj_sign'] * np.inf
        return layout['obj_sign'] * self.scs_output['info']['pobj'] + layout['obj_const']

    def _eval(self, expr: LinearExpr):
        # as cvxpy: no value without a solution
        if self.status not in self.SOLVED:
            return None
        x = self.scs_output['x']
        v = expr.const + sum(coeff * x[idx] for idx, coeff in expr.terms.items())
        return v.item() if isinstance(v, np.generic) else v

    def summary(self):
        assert self.problem is not None and self.status is not None
        data, info = self.data, self.scs_output['info']
        return {
            'n_vars': data.n_vars,
            'n_constraints': len(data.psd_blocks) + int(data.affines_mat.shape[0] > 0) + len(data.energy_ineqs),
            'status': self.status,
            'value': self._value(),
            'objective_sense': data.objective_sense,
            'observables': {
                name: [self._eval(item) for item in obs] if isinstance(obs, list) else self._eval(obs)
                for name, obs in data.observables.items()
            },
            'solver_name': self.name,
            'solve_time': info['solve_time'] / 1000,
            'num_iters': info['iter'],
        }

    @staticmethod
    def _to_scs(data: SDPData, herm_tol: float = 1e-12):
        r'''
            Convert to SCS standard form with native complex PSD cones:

                min c^T x   s.t.   A x + s = b,   s in K = {0}^z x R_+^l x prod_k H_+^{d_k},

            row order z (affines) -> l (energy window) -> cs (PSD blocks).
            Each PSD block M = C + A_k x is entered through its Hermitian part
            H = (M + M^H)/2 as cvec(H) = b_k - A'_k x (lower triangle by column;
            diagonal H_jj, off-diagonal sqrt(2) Re H_ij, sqrt(2) Im H_ij).

            Returns (scs_data, cone, layout):
                scs_data: {'A': csc (m, n), 'b': (m,), 'c': (n,)}, all real float64
                cone: {'z': int, 'l': int, 'cs': [d_k, ...]}
                layout: obj_sign, obj_const   (objective value = obj_sign * c^T x + obj_const)
                      block_offsets, block_dims (row range of block k:
                                                 block_offsets[k] : block_offsets[k] + d_k^2)
                      herm_err (max |M - M^H| entry of the coefficients, per block)

            Raises ValueError if a PSD block is not Hermitian (herm_err > herm_tol).
        '''
        if data.var_cpx:
            raise NotImplementedError('_to_scs: complex moment variables are not supported')
        n = data.n_vars

        def real_part(v, what, tol=1e-12):
            v = np.asarray(v)
            if np.iscomplexobj(v):
                if v.size and np.max(np.abs(v.imag)) > tol:
                    raise ValueError(f'_to_scs: {what} has a non-zero imaginary part')
                v = v.real
            return np.asarray(v, dtype=float)

        def row_of(expr: LinearExpr):
            cols = np.fromiter(expr.terms.keys(), dtype=np.int64, count=len(expr.terms))
            vals = np.array(list(expr.terms.values()), dtype=complex).real
            return cols, vals, complex(expr.const).real

        # ---- objective (real part, as cp.real in SDPSolver)
        cols, vals, obj_const = row_of(data.objective)
        g = np.zeros(n)
        np.add.at(g, cols, vals)
        if data.objective_sense == 'min':
            obj_sign = 1.
        elif data.objective_sense == 'max':
            obj_sign = -1.
        else:
            raise ValueError(f'unknown objective sense: {data.objective_sense}')
        c = obj_sign * g

        A_parts, b_parts = [], []

        # ---- zero cone: affines_mat = [A_aff | a0],  A_aff x + a0 = 0
        aff = sp.sparse.csr_matrix(data.affines_mat)
        n_z = aff.shape[0]
        if n_z:
            if aff.shape[1] != n + 1:
                raise ValueError(f'affines_mat has {aff.shape[1]} columns, expected {n + 1}')
            A_z = aff[:, :n]
            A_z = sp.sparse.csr_matrix((real_part(A_z.data, 'affines_mat'), A_z.indices, A_z.indptr),
                                       shape=A_z.shape)
            A_parts.append(A_z)
            b_parts.append(-real_part(aff[:, n].toarray().ravel(), 'affines_mat'))

        # ---- positive cone: e0 + g^T x >= 0  ->  s = e0 - (-g)^T x
        n_l = len(data.energy_ineqs)
        if n_l:
            r_idx, c_idx, v_all, b_l = [], [], [], np.zeros(n_l)
            for r, ineq in enumerate(data.energy_ineqs):
                cols, vals, const = row_of(ineq)
                r_idx.append(np.full(len(cols), r)); c_idx.append(cols); v_all.append(-vals)
                b_l[r] = const
            A_parts.append(sp.sparse.csr_matrix(
                (np.concatenate(v_all), (np.concatenate(r_idx), np.concatenate(c_idx))), shape=(n_l, n)))
            b_parts.append(b_l)

        # ---- complex PSD cones
        def herm_cvec_maps(d):
            # real maps T_re, T_im (d^2 x d^2) with cvec(Herm(M)) = T_re Re vec(M) + T_im Im vec(M),
            # vec in column-major order (idx = i + j d), cvec as in SCS (lower triangle by column)
            jj, ii = np.triu_indices(d)                 # (j, i) with i >= j, ordered by column j then i
            ij = ii + jj * d                           # vec index of M_ij (lower)
            ji = jj + ii * d                           # vec index of M_ji (upper)
            diag = ii == jj
            # output row of each lower entry: diag -> 1 row, off-diag -> 2 rows (Re, Im)
            width = np.where(diag, 1, 2)
            start = np.concatenate([[0], np.cumsum(width)[:-1]])
            h = 1. / np.sqrt(2.)
            od = ~diag
            re_rows = np.concatenate([start[diag], start[od], start[od]])
            re_cols = np.concatenate([ij[diag], ij[od], ji[od]])
            re_vals = np.concatenate([np.ones(diag.sum()), np.full(od.sum(), h), np.full(od.sum(), h)])
            im_rows = np.concatenate([start[od] + 1, start[od] + 1])
            im_cols = np.concatenate([ij[od], ji[od]])
            im_vals = np.concatenate([np.full(od.sum(), h), np.full(od.sum(), -h)])
            T_re = sp.sparse.csr_matrix((re_vals, (re_rows, re_cols)), shape=(d * d, d * d))
            T_im = sp.sparse.csr_matrix((im_vals, (im_rows, im_cols)), shape=(d * d, d * d))
            return T_re, T_im

        block_dims, block_offsets, herm_err = [], [], []
        row = n_z + n_l
        for k, block in enumerate(data.psd_blocks):
            d = block.dim
            if block.n_vars != n:
                raise ValueError(f'PSD block has n_vars={block.n_vars}, expected {n}')
            A_k, C_k = block.affine_map()              # vec(M) = A_k x + C_k
            A_k = sp.sparse.csr_matrix(A_k, dtype=complex)
            C_k = np.asarray(C_k, dtype=complex)

            # Hermiticity diagnostic: vec(M^H)[i + j d] = conj(M_ji)
            perm = (np.arange(d * d) % d) * d + np.arange(d * d) // d
            dA = A_k - A_k[perm].conj()
            err = max(np.max(np.abs(dA.data), initial=0.), np.max(np.abs(C_k - C_k[perm].conj()), initial=0.))
            if err > herm_tol:
                raise ValueError(f'_to_scs: PSD block {k} (dim {d}) is not Hermitian: '
                                 f'max |M - M^H| coefficient = {err:.1e} > {herm_tol:.0e}')
            herm_err.append(float(err))

            T_re, T_im = herm_cvec_maps(d)
            A_cs = -(T_re @ sp.sparse.csr_matrix(A_k.real) + T_im @ sp.sparse.csr_matrix(A_k.imag))
            b_cs = T_re @ C_k.real + T_im @ C_k.imag
            A_parts.append(A_cs)
            b_parts.append(b_cs)
            block_dims.append(d)
            block_offsets.append(row)
            row += d * d

        if A_parts:
            A = sp.sparse.vstack(A_parts, format='csc')
            b = np.concatenate(b_parts)
        else:
            A = sp.sparse.csc_matrix((0, n))
            b = np.zeros(0)
        A = sp.sparse.csc_matrix(A, dtype=float)
        A.eliminate_zeros()
        A.sort_indices()

        cone = {'z': n_z, 'l': n_l}
        if block_dims:
            cone['cs'] = block_dims

        scs_data = {'A': A, 'b': np.ascontiguousarray(b, dtype=float), 'c': np.ascontiguousarray(c, dtype=float)}
        layout = {
            'obj_sign': obj_sign,
            'obj_const': obj_const,
            'block_offsets': block_offsets,
            'block_dims': block_dims,
            'herm_err': herm_err,
        }
        return scs_data, cone, layout


if __name__ == '__main__':

    from compiler.ising import IsingParams, IsingCompiler, build_basis_reprs

    params=IsingParams(L=16, J=1., h=1.)
    basis0 = ['I', 'X', 'Y', 'Z', 'ZZ']
    basis1 = basis0 + ['XX', 'YY', 'XZ', 'XY', 'YX', 'YZ', 'ZX', 'ZY']
    basis2 = basis0 + ['XX', 'YY', 'XXX', 'YYY', 'ZZZ']

    compiler = IsingCompiler(params=params)
    compiler.compile(
        basis_reprs=build_basis_reprs(params.L, basis1) # basis0, basis1, basis2
    )
    print(compiler.summary())

    solver = SDPSolver()
    solver.build(compiler.sdp_data())
    solver.solve(backend='SCS', eps=1e-4, max_iters=10_000)

    print(solver.problem.value)
    print(solver.problem.status)
    print(solver.summary())

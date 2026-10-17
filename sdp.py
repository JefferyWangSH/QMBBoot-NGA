from dataclasses import dataclass, field
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

    def realify(self) -> list['LinearExpr']:
        r'''
            realification of the equation \sum_k c_k m_k + c_0 = 0: returns the non-empty real and
            imaginary parts \sum_k Re(c_k) m_k + Re(c_0) and \sum_k Im(c_k) m_k + Im(c_0) as real LinearExprs.

            assumes real moment variables m_k (var_cpx = False).
        '''
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

        realify_phase (optional, set by the compiler): integers phi_i such that

            R = D^H M D,   D = diag(i^{phi_i}),

        is real for every m (e.g. blocks at complex-conjugation invariant momenta);
        M >= 0 <=> R >= 0. None means no phase is given and M is used as it is.
        Only diagonal D is considered, because a general unitary transformation
        makes the affine map of R dense.

        block_key and row_keys (optional, set by the compiler) identify blocks and rows across SDPData:
            block_key: symmetry sector of the block (e.g. (momentum, parities)), unique within an SDPData;
            row_keys:  key of the basis operator of each row, unique within the block.
        In two SDPData, entries between rows of equal keys in blocks of equal keys are the same
        affine maps of variables of equal keys (see SDPData.var_keys). None means not given.
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

    # exponents phi_i of the diagonal phase D = diag(i^{phi_i}) making D^H M D real
    realify_phase: list[int] | None = None

    # symmetry sector of the block, unique within an SDPData,
    # and basis operator (string / monomial) key of each row, unique within the block
    block_key: object = None
    row_keys: list | None = None

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

    def affine_map(self, realify: bool = False) -> tuple[sp.sparse.csc_matrix, np.ndarray]:
        '''
            sparse A_{ij,k} of shape (dim^2, n_vars) and dense C_{ij} of shape (dim^2,),
            such that vec(X) = A @ m + C with column-major vec (idx = i + j*dim), where

                X = M                    if not realify,
                X = R = D^H M D (real)   if realify (requires realify_phase).

            R_{ij} = i^{phi_j - phi_i} M_{ij} is exact, so its imaginary part must vanish exactly.
        '''
        if realify and self.realify_phase is None:
            raise ValueError('realify requested but realify_phase is not set')

        shape = (self.dim*self.dim, self.n_vars)

        A = sp.sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=shape)
        dtype = np.result_type(np.asarray(self.vals), np.asarray(self.const_vals), float)
        C = np.zeros(self.dim*self.dim, dtype=dtype)
        np.add.at(C, self.const_rows, self.const_vals)
        if not realify:
            return A, C

        phi = np.asarray(self.realify_phase, dtype=np.int64)
        if phi.shape != (self.dim,):
            raise ValueError(f'realify_phase has shape {phi.shape}, expected ({self.dim},)')

        # exact powers of i, no complex exponentiation
        idx = np.arange(self.dim*self.dim)
        powers = np.array([1, 1j, -1, -1j])
        scale = powers[(phi[idx // self.dim] - phi[idx % self.dim]) % 4]

        A = (sp.sparse.diags(scale) @ A).tocsc()
        C = scale * C
        if np.any(A.data.imag != 0) or np.any(C.imag != 0):
            raise ValueError('D^H M D is not real for the given realify_phase')
        return sp.sparse.csc_matrix(A.real), np.ascontiguousarray(C.real)


@dataclass(slots=True)
class AffineConstraints:
    n_vars: int
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    vals: list[float|complex] = field(default_factory=list)
    n_rows: int = 0

    def add(self, expr: LinearExpr):
        for real_expr in expr.realify():
            for idx, coeff in real_expr.terms.items():
                self.rows.append(self.n_rows)
                self.cols.append(idx)
                self.vals.append(coeff)
            if real_expr.const != 0:
                self.rows.append(self.n_rows)
                self.cols.append(self.n_vars)
                self.vals.append(real_expr.const)
            self.n_rows += 1

    def matrix(self, prune: bool = False, tol: float = 1e-12):
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

    # optional (set by the compiler): hashable key of each variable;
    # the same key in two SDPData means the same moment variable. None means no key is given.
    var_keys: list | None = None


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
        import cvxpy as cp
        res = cp.Constant(expr.const)
        for idx, coeff in expr.terms.items():
            res += coeff * self.vars[idx]
        return res

    def build(self, data: SDPData):
        import cvxpy as cp

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ variables
        if data.var_cpx:
            raise NotImplementedError('build: complex moment variables are not supported')
        self.vars = cp.Variable(data.n_vars, complex=data.var_cpx, name='vars')
        self.constraints = []

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PSD constraints
        self.psd_exprs = []
        for block in data.psd_blocks:
            # M, kept for psd_values()
            A, C = block.affine_map(realify=False)
            expr = cp.reshape(A @ self.vars + cp.Constant(C), (block.dim, block.dim), order='F')
            self.psd_exprs.append(expr)

            # R = D^H M D (real) if a realify phase is given: M >= 0 <=> R >= 0,
            # and a real block is not doubled by cvxpy's complex-to-real embedding
            if block.realify_phase is not None:
                A, C = block.affine_map(realify=True)
                expr = cp.reshape(A @ self.vars + cp.Constant(C), (block.dim, block.dim), order='F')
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

    def solve(self, backend: str = 'SCS', warm_start: dict | None = None, **kwargs):
        '''
            warm_start is currently not supported through cvxpy,
            accepted for interface compatibility with ScsSDPSolver.solve.
        '''
        if self.problem is None:
            raise ValueError('problem has not been built')
        if warm_start:
            raise NotImplementedError('SDPSolver does not support warm_start; use ScsSDPSolver')
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

    def solve(self, backend: str | None = None, warm_start: dict | None = None, **kwargs):
        '''
            `backend` is ignored (interface compatibility). kwargs are SCS settings, translated
            as cvxpy does: eps -> eps_abs = eps_rel = eps; default eps_abs = eps_rel = 1e-5.

            warm_start: export_warm() of a previous solution (possibly of another SDPData),
            mapped onto the built problem by the keys (see _import_warm);
            None for a cold start. SCS's scale is not carried over.
        '''
        import scs
        if self.problem is None:
            raise ValueError('problem has not been built')
        if warm_start is not None and not isinstance(warm_start, dict):
            raise TypeError(f'warm_start must be the dict from export_warm(), got {type(warm_start).__name__}')
        opts = dict(kwargs)
        if 'eps' in opts:
            opts['eps_abs'] = opts['eps_rel'] = opts.pop('eps')
        opts.setdefault('eps_abs', 1e-5)
        opts.setdefault('eps_rel', 1e-5)
        opts.setdefault('verbose', False)
        opts.setdefault('adaptive_diag_scale', 0)  # SCS >= 3.3: ~10x slower with real PSD blocks

        scs_data, cone, layout = self.problem
        solver = scs.SCS(scs_data, cone, **opts)
        if warm_start is None:
            self.scs_output = solver.solve(warm_start=False)
        else:
            x, y, s = self._import_warm(warm_start)
            self.scs_output = solver.solve(warm_start=True, x=x, y=y, s=s)
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
            A, C = block.affine_map(realify=False)
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

    r'''
        PSD block vectorization as in SCS: vec(X) is column-major (idx = i + j d), and the lower
        triangle is ordered by column, off-diagonal entries scaled by sqrt(2):

            svec(X) = T vec(X)                              = svec(Sym X),   d (d+1) / 2 rows
            cvec(X) = T_re Re vec(X) + T_im Im vec(X)       = cvec(Herm X),  d^2 rows,
                      off-diagonal (sqrt(2) Re H_ij, sqrt(2) Im H_ij)

        smat and cmat are the adjoint maps T^T and T_re^T + i T_im^T, which invert svec and cvec
        on symmetric and Hermitian matrices: svec(smat(u)) = u, cvec(cmat(u)) = u, and
        <smat(u), X> = u^T svec(X), Re <cmat(u), X> = u^T cvec(X).
    '''

    @staticmethod
    def _svec_map(d):
        # real map T (d(d+1)/2 x d^2)
        jj, ii = np.triu_indices(d)                 # (j, i) with i >= j, ordered by column j then i
        ij = ii + jj * d
        ji = jj + ii * d
        diag = ii == jj
        od = ~diag
        rows = np.arange(len(ii))
        h = 1. / np.sqrt(2.)
        t_rows = np.concatenate([rows[diag], rows[od], rows[od]])
        t_cols = np.concatenate([ij[diag], ij[od], ji[od]])
        t_vals = np.concatenate([np.ones(diag.sum()), np.full(od.sum(), h), np.full(od.sum(), h)])
        return sp.sparse.csr_matrix((t_vals, (t_rows, t_cols)), shape=(len(ii), d * d))

    @staticmethod
    def _cvec_maps(d):
        # real maps T_re, T_im (d^2 x d^2)
        jj, ii = np.triu_indices(d)                # (j, i) with i >= j, ordered by column j then i
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

    @staticmethod
    def _svec(v, d):
        '''
            svec(Sym X) from real v = vec(X) of shape (d^2,) or (d^2, k), dense or sparse
        '''
        return ScsSDPSolver._svec_map(d) @ v

    @staticmethod
    def _cvec(v, d):
        '''
            cvec(Herm X) from v = vec(X) of shape (d^2,) or (d^2, k), dense or sparse
        '''
        T_re, T_im = ScsSDPSolver._cvec_maps(d)
        return T_re @ v.real + T_im @ v.imag

    @staticmethod
    def _smat(u, d):
        '''
            real symmetric (d, d) matrix X with svec(X) = u
        '''
        return (ScsSDPSolver._svec_map(d).T @ u).reshape((d, d), order='F')

    @staticmethod
    def _cmat(u, d):
        '''
            Hermitian (d, d) matrix X with cvec(X) = u
        '''
        T_re, T_im = ScsSDPSolver._cvec_maps(d)
        return (T_re.T @ u + 1j * (T_im.T @ u)).reshape((d, d), order='F')

    @staticmethod
    def _to_scs(data: SDPData, herm_tol: float = 1e-12):
        r'''
            Convert to SCS standard form with real and native complex PSD cones:

                min c^T x   s.t.   A x + s = b,
                s in K = {0}^z x R_+^l x prod_k S_+^{d_k} x prod_k H_+^{d_k},

            row order z (affines) -> l (energy window) -> s (real PSD blocks) -> cs (complex PSD blocks).
            A block X = C + A_k x is X = R = D^H M D if realify_phase is set, otherwise X = M.
            X is real if all its coefficients are real (then cone s), otherwise complex (cone cs).
            X is entered through its Hermitian part H = (X + X^H)/2, lower triangle by column:
                s:  svec(H), diagonal H_jj, off-diagonal sqrt(2) H_ij                 (d (d+1) / 2 rows)
                cs: cvec(H), diagonal H_jj, off-diagonal sqrt(2) Re H_ij, sqrt(2) Im H_ij  (d^2 rows)

            Returns (scs_data, cone, layout):
                scs_data: {'A': csc (m, n), 'b': (m,), 'c': (n,)}, all real float64
                cone: {'z': int, 'l': int, 's': [d_k, ...], 'cs': [d_k, ...]}
                layout: obj_sign, obj_const   (objective value = obj_sign * c^T x + obj_const)
                      block_cones, block_dims, block_offsets   (per block k in data.psd_blocks order:
                                 cone 's' or 'cs', and row range block_offsets[k] : + rows of the cone)
                      herm_err (max |X - X^H| entry of the coefficients, per block)

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

        # ---- PSD cones
        psd_parts = {'s': [], 'cs': []}            # per cone: (k, A rows, b rows)
        n_blocks = len(data.psd_blocks)
        block_cones, block_dims, herm_err = [None] * n_blocks, [None] * n_blocks, [None] * n_blocks
        for k, block in enumerate(data.psd_blocks):
            d = block.dim
            if block.n_vars != n:
                raise ValueError(f'PSD block has n_vars={block.n_vars}, expected {n}')
            # vec(X) = A_k x + C_k,  X = R if realify_phase is set, else M
            A_k, C_k = block.affine_map(realify=block.realify_phase is not None)
            A_k = sp.sparse.csr_matrix(A_k, dtype=complex)
            C_k = np.asarray(C_k, dtype=complex)

            # Hermiticity diagnostic: vec(X^H)[i + j d] = conj(X_ji)
            perm = (np.arange(d * d) % d) * d + np.arange(d * d) // d
            dA = A_k - A_k[perm].conj()
            err = max(np.max(np.abs(dA.data), initial=0.), np.max(np.abs(C_k - C_k[perm].conj()), initial=0.))
            if err > herm_tol:
                raise ValueError(f'_to_scs: PSD block {k} (dim {d}) is not Hermitian: '
                                 f'max |X - X^H| coefficient = {err:.1e} > {herm_tol:.0e}')
            herm_err[k] = float(err)
            block_dims[k] = d

            if np.all(A_k.data.imag == 0) and np.all(C_k.imag == 0):
                block_cones[k] = 's'
                psd_parts['s'].append((k, -ScsSDPSolver._svec(A_k.real, d), ScsSDPSolver._svec(C_k.real, d)))
            else:
                block_cones[k] = 'cs'
                psd_parts['cs'].append((k, -ScsSDPSolver._cvec(A_k, d), ScsSDPSolver._cvec(C_k, d)))

        # real PSD cones before complex PSD cones (SCS cone order)
        block_offsets = [None] * n_blocks
        row = n_z + n_l
        for kind in ('s', 'cs'):
            for k, A_blk, b_blk in psd_parts[kind]:
                A_parts.append(A_blk)
                b_parts.append(b_blk)
                block_offsets[k] = row
                row += A_blk.shape[0]

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
        for kind in ('s', 'cs'):
            if psd_parts[kind]:
                cone[kind] = [block_dims[k] for k, _, _ in psd_parts[kind]]

        scs_data = {'A': A, 'b': np.ascontiguousarray(b, dtype=float), 'c': np.ascontiguousarray(c, dtype=float)}
        layout = {
            'obj_sign': obj_sign,
            'obj_const': obj_const,
            'block_cones': block_cones,
            'block_dims': block_dims,
            'block_offsets': block_offsets,
            'herm_err': herm_err,
        }
        return scs_data, cone, layout

    def export_warm(self) -> dict:
        '''
            basis-independent warm start of the current solution, keyed by the compiler keys
            (SDPData.var_keys, PSDConstraints.block_key and row_keys):

                x:      {var key: x_i}
                y_l:    energy-window duals, shape (n_l,)
                y_psd:  {block_key: (row_keys, Y)},  Y = D Y_X D^H the PSD dual in the M basis,
                        with Y_X the dual of the cone matrix X (R = D^H M D, or M with D = I)

            Not exported: s (recomputed from x), zero-cone duals (rows selected by SPQR
            differ between problems), and SCS's adapted scale.
            Raises ValueError without a solution or without the keys.
        '''
        if self.status not in self.SOLVED:
            raise ValueError(f'no solution available (status: {self.status})')
        data = self.data
        if data.var_keys is None or len(data.var_keys) != data.n_vars:
            raise ValueError('export_warm: SDPData.var_keys is not set (one key per variable)')
        if len(set(data.var_keys)) != data.n_vars:
            raise ValueError('export_warm: SDPData.var_keys are not unique')

        _, cone, layout = self.problem
        y = self.scs_output['y']
        n_z, n_l = cone['z'], cone['l']

        y_psd = {}
        for k, block in enumerate(data.psd_blocks):
            if block.block_key is None or block.row_keys is None:
                raise ValueError(f'export_warm: PSD block {k} has no block_key or row_keys')
            if block.block_key in y_psd:
                raise ValueError(f'export_warm: PSD block key {block.block_key!r} is not unique')
            if len(block.row_keys) != block.dim or len(set(block.row_keys)) != block.dim:
                raise ValueError(f'export_warm: PSD block {k} needs {block.dim} unique row_keys')

            d, off = block.dim, layout['block_offsets'][k]
            if layout['block_cones'][k] == 's':
                Y = self._smat(y[off:off + d * (d + 1) // 2], d)
            else:
                Y = self._cmat(y[off:off + d * d], d)
            if block.realify_phase is not None:
                # Y_M = D Y_R D^H, D = diag(i^phi) with exact powers of i
                w = np.array([1, 1j, -1, -1j])[np.mod(block.realify_phase, 4)]
                Y = w[:, None] * Y * w.conj()[None, :]
            y_psd[block.block_key] = (list(block.row_keys), Y)

        return {
            'x': dict(zip(data.var_keys, self.scs_output['x'].tolist())),
            'y_l': y[n_z:n_z + n_l].copy(),
            'y_psd': y_psd,
        }

    def _import_warm(self, warm: dict):
        '''
            SCS warm start (x, y, s) on the built problem from export_warm() of a previous solution
            (possibly of another SDPData), matched by the keys:

                x: shared variables copied, new variables 0
                y: zero cone 0; energy window copied if of the same size, else 0;
                   PSD block: principal submatrix of the old dual Y (same block_key) on the shared rows,
                   zero for new rows and new blocks, then Y_X = D^H Y D (real part in cone s)
                s: Pi_K(b - A x), the s in K closest to b - A x (smallest primal residual for x):
                   zero cone 0; energy window max(b - A x, 0); PSD block Proj_PSD(Herm X(x))

            y and s lie in the dual and primal cone respectively.
        '''
        if self.problem is None:
            raise ValueError('problem has not been built')
        data = self.data
        if data.var_keys is None or len(data.var_keys) != data.n_vars:
            raise ValueError('_import_warm: SDPData.var_keys is not set (one key per variable)')

        scs_data, cone, layout = self.problem
        n_z, n_l = cone['z'], cone['l']

        x = np.array([warm['x'].get(key, 0.) for key in data.var_keys], dtype=float)
        r = scs_data['b'] - scs_data['A'] @ x       # vec_K(Herm X(x)) on the PSD rows
        y = np.zeros_like(r)
        s = np.zeros_like(r)

        if len(warm['y_l']) == n_l:
            y[n_z:n_z + n_l] = warm['y_l']
        s[n_z:n_z + n_l] = np.maximum(r[n_z:n_z + n_l], 0.)

        for k, block in enumerate(data.psd_blocks):
            if block.block_key is None or block.row_keys is None or len(block.row_keys) != block.dim:
                raise ValueError(f'_import_warm: PSD block {k} has no block_key or row_keys')
            d, off = block.dim, layout['block_offsets'][k]
            real = layout['block_cones'][k] == 's'
            rows = slice(off, off + (d * (d + 1) // 2 if real else d * d))

            # y: old dual (M basis) on the shared rows, zero for new rows
            Y = np.zeros((d, d), dtype=complex)
            old = warm['y_psd'].get(block.block_key)
            if old is not None:
                old_keys, Y_old = old
                pos = {key: i for i, key in enumerate(old_keys)}
                new_idx = [a for a, key in enumerate(block.row_keys) if key in pos]
                if new_idx:
                    old_idx = [pos[block.row_keys[a]] for a in new_idx]
                    Y[np.ix_(new_idx, new_idx)] = Y_old[np.ix_(old_idx, old_idx)]
            if block.realify_phase is not None:
                # Y_R = D^H Y_M D, D = diag(i^phi) with exact powers of i
                w = np.array([1, 1j, -1, -1j])[np.mod(block.realify_phase, 4)]
                Y = w.conj()[:, None] * Y * w[None, :]

            # s: projection of Herm X(x) onto the PSD cone
            H = self._smat(r[rows], d) if real else self._cmat(r[rows], d)
            lam, V = np.linalg.eigh(H)
            S = (V * np.maximum(lam, 0.)) @ V.conj().T

            if real:
                y[rows] = self._svec(Y.real.reshape(-1, order='F'), d)
                s[rows] = self._svec(S.real.reshape(-1, order='F'), d)
            else:
                y[rows] = self._cvec(Y.reshape(-1, order='F'), d)
                s[rows] = self._cvec(S.reshape(-1, order='F'), d)

        return x, y, s


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

from dataclasses import dataclass
import numpy as np
import scipy as sp

from compiler.core import HeisenbergCompilerCore
from operators.pauli import PauliString, PauliOperator
from sdp import AffineExpr, PSDConstraints, AffineConstraints, SDPData


@dataclass(slots=True)
class HeisenbergParams:
    L: int = 8
    J1: float = 1.
    J2: float = 1.


def _two_site_pstr(L: int, pauli: str, dist: int):
    return PauliString.from_str(pauli + 'I'*(dist-1) + pauli + 'I'*(L-dist-1))


def build_hamil(params: HeisenbergParams):
    assert params.L >= 3
    hamil_op = PauliOperator()

    for pauli in 'XYZ':
        nn = _two_site_pstr(params.L, pauli, 1)
        nnn = _two_site_pstr(params.L, pauli, 2)
        for shift in range(params.L):
            hamil_op.add(nn.translate(shift), .25 * params.J1 / params.L)
            hamil_op.add(nnn.translate(shift), .25 * params.J2 / params.L)

    return hamil_op


def build_szz(params: HeisenbergParams, r: int):
    assert 0 <= r <= params.L//2
    pstr = PauliString.from_str('I'*params.L) if r == 0 else _two_site_pstr(params.L, 'Z', r)
    return PauliOperator({pstr: .25})


class HeisenbergCompiler:
    L: int
    params: HeisenbergParams

    basis_reprs: list[PauliString]

    _core: HeisenbergCompilerCore

    var_cpx: bool = False
    vars: list[PauliString]
    _moments_lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None

    ward_ops: dict[str, int]

    # we divide PSD blocks using translation and pi-angle spin-rotation symmetry
    block_reprs: list[list[PauliString]]
    block_momenta: list[int]
    block_keys: list[tuple[int, int]]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: PauliOperator

    # SDP objective
    obj_op: PauliOperator
    obj_expr: AffineExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[AffineExpr]

    # uncertified observables
    obs_ops: dict[str, PauliOperator | list[PauliOperator]]
    obs_exprs: dict[str, AffineExpr | list[AffineExpr]]

    def __init__(self,
        params: HeisenbergParams,
        *,
        obj_op: PauliOperator | None = None,
        obj_sense: str = 'min',
        e_lb: float = None,
        e_ub: float = None,
        obs_ops: dict | None = None,
    ):
        self.L = params.L
        self.params = params
        self.ward_ops = {'hamil': 0, 'Sx': 0, 'Sy': 0, 'Sz': 0}

        self.hamil_op = build_hamil(params)

        # the C++ core takes over:
        #   1) building the SDP variables and the moments lookup (OpenMP),
        #   2) generating the Ward identities (OpenMP),
        #   3) generating the descendants,
        #   4) compiling Pauli operators into linear expressions of the SDP variables.
        # the Hamiltonian coefficients are scaled from energy density to total H.
        self._core = HeisenbergCompilerCore(
            L=self.L,
            hamil_paulis=list(self.hamil_op.terms),
            hamil_coeffs=[self.L * h_coeff for h_coeff in self.hamil_op.terms.values()],
        )

        if obj_sense not in ('min', 'max'):
            raise ValueError('objective sense must be min or max')
        self.obj_sense = obj_sense

        self.obj_op = self.hamil_op if obj_op is None else obj_op
        if self.obj_op != self.hamil_op:
            if e_lb is None or e_ub is None:
                raise ValueError('observable objective requires energy bounds e_lb and e_ub')
            self.e_lb = e_lb
            self.e_ub = e_ub
        else:
            if self.obj_sense != 'min':
                raise ValueError("Hamiltonian objective requires obj_sense='min'")
            if e_lb is not None or e_ub is not None:
                raise ValueError('e_lb and e_ub are only valid for non-Hamiltonian objectives')
            self.e_lb = None
            self.e_ub = None

        self.obs_ops = {} if obs_ops is None else obs_ops.copy()

    def trans_canon(self, pstr: PauliString) -> int:
        return pstr.trans_canon

    def trans_canon_rep(self, pstr: PauliString) -> PauliString:
        return pstr.trans_canon_rep

    def _build_vars(self):
        r'''
            build SDP variables in one pass over (O_b, O_a, r), and then keep the lookup
            (b, a, r) -> (var_idx, coeff) with (T_r O_a)^\dag O_b = coeff * m[var_idx] in
            self._moments_lookup = (pair_offsets, pair_sizes, rs, idxs, coeffs)
            for later use in _build_psd, which only adds the Fourier phases.

            for each pair (O_a, O_b) we record a list of terms (r, var_idx, coeff), one per translation r:

                (T_r O_a)^\dag O_b = coeff * m[var_idx],   coeff = +-1 or +-i;

            an r whose product vanishes by symmetry gives no term.
            Only the pairs a >= b (lower triangle) are recorded because M(q) is Hermitian.

            This is a 3d array [b, a, r] with the missing r removed, flattened to 1d (as in CSR):
            the terms of all pairs are concatenated in the order (b, a, r) into three 1d arrays of the same length,

                rs[t], idxs[t], coeffs[t]   (translation, variable index, coefficient),

            and the terms of the pair (a, b) are t in [pair_offsets[k], pair_offsets[k] + pair_sizes[k]),
            k = b*n + a, n = len(basis_reprs); pair_offsets and pair_sizes have length n*n,
            and the pairs a < b have size 0.

            This is done by the C++ core with OpenMP:
            self._core.build_vars clears and refills the variables and their lookup
            inside the core in place, and returns the moments lookup.
        '''
        self._moments_lookup = self._core.build_vars(self.basis_reprs)
        self.vars = self._core.vars()

    def fourier_phase(self, q: int, r: int) -> complex:
        # q is the integer momentum-sector index;
        # exact at multiples of L/4 and n -> -n symmetric, so Fourier sums cancel exactly
        n = (q * r) % self.L
        if 4 * n % self.L == 0:
            return 1j ** (4 * n // self.L)
        if 2 * n > self.L:
            n -= self.L
        return np.exp(1j * 2 * np.pi * n / self.L)

    @staticmethod
    def nonzero_fourier(pstr: PauliString, q: int) -> bool:
        return (q * pstr.period) % pstr.L == 0

    def _build_block_reprs(self):
        self.block_reprs = []
        self.block_momenta = []
        self.block_keys = [] # symmetry sector (q, pi-rotation charge)

        for q in range(self.L//2 + 1): # K symmetry
            charge_reprs = {}
            for pstr in self.basis_reprs:
                if not self.nonzero_fourier(pstr, q):
                    continue
                charge = pstr.pi_rot_charge()
                charge_reprs.setdefault(charge, []).append(pstr)

            for charge in sorted(charge_reprs):
                self.block_reprs.append(charge_reprs[charge])
                self.block_momenta.append(q)
                self.block_keys.append((q, charge))

    def _build_psd(self):
        r'''
            build momentum PSD blocks M(k) from self._moments_lookup of _build_vars:
            M_{ab}(q) = \sum_r e^{iqr} / L * coeff * m[var_idx] over the list of the pair (a, b),
            for the lower triangle only
        '''
        self.psd_blocks = []
        n_basis = len(self.basis_reprs)
        n_vars = len(self.vars)
        pair_offsets, pair_sizes, r_terms, var_terms, coeff_terms = self._moments_lookup
        basis_index = {pstr.mask: a for a, pstr in enumerate(self.basis_reprs)}

        for q, block_key, block_basis in zip(self.block_momenta, self.block_keys, self.block_reprs):
            dim = len(block_basis)
            psd = PSDConstraints(
                n_vars=n_vars,
                dim=dim,
                block_key=block_key,
                basis_keys=[self.trans_canon(pstr) for pstr in block_basis],
            )

            phases = np.array([self.fourier_phase(q, r) / self.L for r in range(self.L)])
            # block_basis keeps the order of basis_reprs, so the lower triangle (row >= col) has a >= b
            block_index = np.array([basis_index[pstr.mask] for pstr in block_basis])
            if np.any(np.diff(block_index) <= 0):
                raise ValueError('block_basis must follow the order of basis_reprs without duplicates')
            rows, cols = np.tril_indices(dim)
            pairs = block_index[cols] * n_basis + block_index[rows]
            starts = pair_offsets[pairs]
            counts = pair_sizes[pairs]
            terms = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)

            # COO terms of the affine map vec(M(q)) = A @ m, lower triangle
            psd.rows = np.repeat(rows + cols*dim, counts)
            psd.cols = var_terms[terms]
            psd.vals = phases[r_terms[terms]] * coeff_terms[terms]

            # at complex-conjugation invariant momenta (q = -q mod L),
            # K maps M(k) to itself and D^H M(k) D is real for D = diag(i^{n_a}),
            # n_a = parity of the number of Y in P_a
            if (2 * q) % self.L == 0:
                psd.realify_phase = np.array([pstr.parity() for pstr in block_basis], dtype=np.int64)
            self.psd_blocks.append(psd)

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        id_expr = self._compile_expr(PauliOperator({PauliString.identity(self.L): 1}))
        if id_expr is None:
            raise ValueError('current basis cannot represent the identity operator')
        id_expr.const = -1
        self.affines.add(id_expr)

        # Ward identities <[C, O]> == 0 for each charge C in ward_ops:
        # the stationarity ones (hamil) and the SO(3) ones (Sx, Sy, Sz).
        for charge in self.ward_ops:
            offsets, idxs, coeffs = self._core.build_ward_identities(charge)
            self.affines.add_batch(offsets, idxs, coeffs)
            self.ward_ops[charge] = len(offsets) - 1

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _compile_expr(self, op: PauliOperator) -> AffineExpr | None:
        # None if a term cannot be represented by an SDP variable
        expr = self._core.compile_expr(list(op.terms), list(op.terms.values()))
        if expr is None:
            return None
        idxs, coeffs = expr
        return AffineExpr(terms=dict(zip(idxs.tolist(), coeffs.tolist())), const=0)

    def compile(self, basis_reprs: list[PauliString]):
        self.basis_reprs = list(basis_reprs)
        for pstr in self.basis_reprs:
            if pstr.L != self.L:
                raise ValueError('Pauli string length inconsistent with system size L')

        self._build_vars()

        self.obj_expr = self._compile_expr(self.obj_op)
        if self.obj_expr is None:
            raise ValueError('current basis cannot represent the SDP objective')

        self.energy_ineqs = []
        hamil_expr = self._compile_expr(self.hamil_op)
        if hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        if self.e_lb is not None:
            self.energy_ineqs.append(AffineExpr(
                terms=hamil_expr.terms.copy(),
                const=hamil_expr.const - self.e_lb,
            ))
        if self.e_ub is not None:
            self.energy_ineqs.append(AffineExpr(
                terms={idx: -coeff for idx, coeff in hamil_expr.terms.items()},
                const=self.e_ub - hamil_expr.const,
            ))

        self.obs_exprs = {}
        for name, obs in self.obs_ops.items():
            if isinstance(obs, list):
                exprs = []
                for idx, op in enumerate(obs):
                    expr = self._compile_expr(op)
                    if expr is None:
                        raise ValueError(f'current basis cannot represent observable: {name}[{idx}] = {op}')
                    exprs.append(expr)
                self.obs_exprs[name] = exprs
            else:
                expr = self._compile_expr(obs)
                if expr is None:
                    raise ValueError(f'current basis cannot represent observable: {name} = {obs}')
                self.obs_exprs[name] = expr

        self._build_block_reprs()
        self._build_psd()
        self._moments_lookup = None
        self._build_affines()

    def descendants(self, pstr: PauliString):
        r'''
            calculate

                [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)

            as a list [(O_\mu(0), s, coeff), ...]
        '''
        return self._core.descendants(pstr)

    def _get_expr_str(self, expr: AffineExpr) -> str:
        parts = [
            f'{coeff}*<{str(self.vars[idx])}>'
            for idx, coeff in expr.terms.items()
        ]
        if expr.const != 0:
            parts.append(str(expr.const))
        return ' + '.join(parts)

    def summary(self):
        return {
            'params': self.params,
            'basis_reprs': len(self.basis_reprs),
            'vars': len(self.vars),
            'ward_ops': self.ward_ops,
            'psd_blocks': len(self.psd_blocks),
            'psd_dims_sum': sum(psd.dim for psd in self.psd_blocks),
            'psd_dims': [psd.dim for psd in self.psd_blocks],
            'affines_raw': self.affines.n_rows,
            'affines_rank': self.affines_mat.shape[0],
            'obj_expr': self._get_expr_str(self.obj_expr),
            'obj_sense': self.obj_sense,
            'obs_exprs': {
                name: [self._get_expr_str(item) for item in expr] if isinstance(expr, list) else self._get_expr_str(expr)
                for name, expr in self.obs_exprs.items()
            },
            'energy_ineqs': [self._get_expr_str(ineq) for ineq in self.energy_ineqs],
        }

    def sdp_data(self):
        return SDPData(
            var_cpx = self.var_cpx,
            n_vars = len(self.vars),
            objective = self.obj_expr,
            objective_sense = self.obj_sense,
            psd_blocks = self.psd_blocks,
            affines_mat = self.affines_mat,
            observables = self.obs_exprs,
            energy_ineqs = self.energy_ineqs,
            var_keys = [var.mask for var in self.vars],
        )

    def clone(self):
        return type(self)(
            self.params,
            obj_op=self.obj_op,
            obj_sense=self.obj_sense,
            e_lb=self.e_lb,
            e_ub=self.e_ub,
            obs_ops=self.obs_ops,
        )


def build_basis_reprs(L: int, basis: list[str]) -> list[PauliString]:
    basis_reprs = []
    for pstr in basis:
        if len(pstr) > L:
            raise ValueError('Pauli string length exceeds system size L')
        basis_reprs.append(PauliString.from_str(pstr + 'I'*(L - len(pstr))))
    return basis_reprs


if __name__ == '__main__':

    params = HeisenbergParams(L=8, J1=1., J2=1.)
    compiler = HeisenbergCompiler(params)
    compiler.compile(build_basis_reprs(params.L, ['I', 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XXX', 'YYY', 'ZZZ']))

    print(*compiler.basis_reprs)
    print(compiler.summary())

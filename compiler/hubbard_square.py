from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from compiler.core import HubbardSquareCompilerCore
from operators.majorana_square import MajoranaMonomialSquare, MajoranaSquareOperator
from sdp import AffineExpr, PSDConstraints, AffineConstraints, SDPData


@dataclass(slots=True)
class HubbardSquareParams:
    Lx: int = 4
    Ly: int = 4
    t: float = 1.0
    U: float = 4.0
    n_particles: int | None = None

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly


def build_hamil(params: HubbardSquareParams) -> MajoranaSquareOperator:
    assert params.Lx >= 2 and params.Ly >= 2
    Lx = params.Lx
    Ly = params.Ly
    n_sites = params.n_sites
    hamil_op = MajoranaSquareOperator()

    for x in range(Lx):
        for y in range(Ly):
            for xp, yp in (((x + 1) % Lx, y), (x, (y + 1) % Ly)):
                for spin in ('u', 'd'):
                    monomial, sign = MajoranaMonomialSquare.from_str(
                        Lx, Ly, f'({x},{y}){spin}+ ({xp},{yp}){spin}-', sign=True
                    )
                    hamil_op.add(monomial, -.5j * params.t * sign / n_sites)

                    monomial, sign = MajoranaMonomialSquare.from_str(
                        Lx, Ly, f'({x},{y}){spin}- ({xp},{yp}){spin}+', sign=True
                    )
                    hamil_op.add(monomial, .5j * params.t * sign / n_sites)

            monomial, sign = MajoranaMonomialSquare.from_str(
                Lx, Ly, f'({x},{y})u+ ({x},{y})u- ({x},{y})d+ ({x},{y})d-', sign=True
            )
            hamil_op.add(monomial, -.25 * params.U * sign / n_sites)

    return hamil_op


def build_number(params: HubbardSquareParams, spin: str | None = None) -> MajoranaSquareOperator:
    assert spin in (None, 'u', 'd')
    Lx = params.Lx
    Ly = params.Ly
    n_sites = params.n_sites
    spins = ('u', 'd') if spin is None else (spin,)
    number_op = MajoranaSquareOperator()
    for s in spins:
        number_op.add(MajoranaMonomialSquare.identity(Lx, Ly), .5)
        for x in range(Lx):
            for y in range(Ly):
                monomial, sign = MajoranaMonomialSquare.from_str(
                    Lx, Ly, f'({x},{y}){s}+ ({x},{y}){s}-', sign=True
                )
                number_op.add(monomial, .5j * sign / n_sites)
    return number_op


def build_sz(params: HubbardSquareParams, x: int, y: int) -> MajoranaSquareOperator:
    assert 0 <= x < params.Lx and 0 <= y < params.Ly
    sz_op = MajoranaSquareOperator()
    for spin, coeff in (('u', .25j), ('d', -.25j)):
        monomial, sign = MajoranaMonomialSquare.from_str(
            params.Lx, params.Ly, f'({x},{y}){spin}+ ({x},{y}){spin}-', sign=True
        )
        sz_op.add(monomial, coeff * sign)
    return sz_op


def build_szz(params: HubbardSquareParams, dx: int, dy: int) -> MajoranaSquareOperator:
    assert 0 <= dx < params.Lx and 0 <= dy < params.Ly
    return build_sz(params, 0, 0).mul(build_sz(params, dx, dy))


def build_double_occ(params: HubbardSquareParams) -> MajoranaSquareOperator:
    Lx = params.Lx
    Ly = params.Ly
    n_sites = params.n_sites
    double_occ_op = MajoranaSquareOperator({MajoranaMonomialSquare.identity(Lx, Ly): .25})
    for x in range(Lx):
        for y in range(Ly):
            for spin in ('u', 'd'):
                monomial, sign = MajoranaMonomialSquare.from_str(
                    Lx, Ly, f'({x},{y}){spin}+ ({x},{y}){spin}-', sign=True
                )
                double_occ_op.add(monomial, .25j * sign / n_sites)

            monomial, sign = MajoranaMonomialSquare.from_str(
                Lx, Ly, f'({x},{y})u+ ({x},{y})u- ({x},{y})d+ ({x},{y})d-', sign=True
            )
            double_occ_op.add(monomial, -.25 * sign / n_sites)
    return double_occ_op


class HubbardSquareCompiler:
    params: HubbardSquareParams
    Lx: int
    Ly: int
    n_sites: int
    n_particles: int | None

    basis_reprs: list[MajoranaMonomialSquare]
    block_reprs: list[list[MajoranaMonomialSquare]]
    block_momenta: list[tuple[int, int]]
    block_keys: list[tuple[tuple[int, int], tuple[int, int], int | None]]

    _core: HubbardSquareCompilerCore

    var_cpx: bool = False
    vars: list[MajoranaMonomialSquare]
    _moments_lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None

    ward_ops: dict[str, int]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaSquareOperator | None
    number_op: MajoranaSquareOperator | None

    obj_op: MajoranaSquareOperator | None
    obj_expr: AffineExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[AffineExpr]

    obs_ops: dict[str, MajoranaSquareOperator | list[MajoranaSquareOperator]]
    obs_exprs: dict[str, AffineExpr | list[AffineExpr]]

    def __init__(
        self,
        params: HubbardSquareParams,
        obj_op: MajoranaSquareOperator | None = None,
        obj_sense: str = 'min',
        e_lb: float | None = None,
        e_ub: float | None = None,
        obs_ops: dict[str, MajoranaSquareOperator | list[MajoranaSquareOperator]] | None = None,
    ):
        if params.Lx <= 0 or params.Ly <= 0:
            raise ValueError('Lx and Ly must be positive')
        if params.n_particles is not None and not (0 <= params.n_particles <= 2 * params.n_sites):
            raise ValueError('n_particles must be between 0 and 2*Lx*Ly')

        self.params = params
        self.Lx = params.Lx
        self.Ly = params.Ly
        self.n_sites = params.n_sites
        self.n_particles = params.n_particles

        self.use_ph_sym = self.Lx % 2 == 0 and self.Ly % 2 == 0 and self.n_particles in (None, self.n_sites)
        self.use_eta_sym = self.use_ph_sym
        self.ward_ops = {'hamil': 0, 'Nu': 0, 'Nd': 0, 'S+': 0}
        if self.use_eta_sym:
            self.ward_ops['eta+'] = 0

        self.hamil_op = build_hamil(params)
        self.number_op = build_number(params)

        # the C++ core takes over:
        #   1) building the SDP variables and the moments lookup (OpenMP),
        #   2) generating the Ward identities (OpenMP),
        #   3) generating the descendants,
        #   4) compiling Majorana operators into linear expressions of the SDP variables.
        # the Hamiltonian coefficients are scaled from energy density to total H.
        self._core = HubbardSquareCompilerCore(
            Lx=self.Lx,
            Ly=self.Ly,
            hamil_monomials=list(self.hamil_op.terms),
            hamil_coeffs=[self.n_sites * h_coeff for h_coeff in self.hamil_op.terms.values()],
            use_ph_sym=self.use_ph_sym,
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

    def trans_canon(self, monomial: MajoranaMonomialSquare) -> int:
        return monomial.trans_canon

    def trans_canon_rep(self, monomial: MajoranaMonomialSquare) -> MajoranaMonomialSquare:
        return monomial.trans_canon_rep

    def fourier_phase(self, q: tuple[int, int], r: tuple[int, int]) -> complex:
        # q is the integer momentum-sector index;
        # phase = w^n with w = e^{2 pi i / N}, N = Lx * Ly, n = qx x Ly + qy y Lx (mod N);
        # exact at multiples of N/4 and n -> -n symmetric, so Fourier sums cancel exactly
        qx, qy = q
        x, y = r
        N = self.Lx * self.Ly
        n = (qx * x * self.Ly + qy * y * self.Lx) % N
        if 4 * n % N == 0:
            return 1j ** (4 * n // N)
        if 2 * n > N:
            n -= N
        return np.exp(1j * 2 * np.pi * n / N)

    @staticmethod
    def nonzero_fourier(monomial: MajoranaMonomialSquare, q: tuple[int, int]) -> bool:
        qx, qy = q
        for (dx, dy), trans_sign in monomial.trans_stabilizer():
            phase_num = qx * dx * monomial.Ly + qy * dy * monomial.Lx
            if trans_sign == 1:
                if phase_num % monomial.n_sites != 0:
                    return False
            else:
                if (2 * phase_num) % (2 * monomial.n_sites) != monomial.n_sites:
                    return False
        return True

    def _compile_expr(self, op: MajoranaSquareOperator) -> AffineExpr | None:
        # None if a term cannot be represented by an SDP variable
        expr = self._core.compile_expr(list(op.terms), list(op.terms.values()))
        if expr is None:
            return None
        idxs, coeffs = expr
        return AffineExpr(terms=dict(zip(idxs.tolist(), coeffs.tolist())), const=0)

    def _build_vars(self):
        r'''
            build SDP variables in one pass over (O_b, O_a, r), r = dx*Ly + dy the translation by (dx, dy),
            and then keep the lookup (b, a, r) -> (var_idx, coeff) with (T_r O_a)^\dag O_b = coeff * m[var_idx] in
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

    def _build_block_reprs(self):
        self.block_reprs = []
        self.block_momenta = []
        self.block_keys = [] # symmetry sector (q, fermion parity, particle-hole parity or None)
        for nx in range(self.Lx):
            for ny in range(self.Ly):
                # complex-conjugation K symmetry makes k and -k equivalent
                # up to complex conjugation and a unitary transformation
                q = (nx, ny)
                neg_q = ((-nx) % self.Lx, (-ny) % self.Ly)
                if q > neg_q:
                    continue

                f_parity_reprs = {}
                for monomial in self.basis_reprs:
                    if not self.nonzero_fourier(monomial, q):
                        continue
                    f_parity = monomial.fermion_parity(spin=True)
                    f_parity_reprs.setdefault(f_parity, []).append(monomial)

                for f_parity in sorted(f_parity_reprs):
                    # for odd total degree, particle-hole symmetry shifts momentum by (pi, pi),
                    # hence relating associated momentum blocks at k and k + (pi, pi)
                    if self.use_ph_sym and f_parity in ((0, 1), (1, 0)):
                        ph_q = ((nx + self.Lx//2) % self.Lx, (ny + self.Ly//2) % self.Ly)
                        neg_ph_q = ((-ph_q[0]) % self.Lx, (-ph_q[1]) % self.Ly)
                        if q != min(q, neg_q, ph_q, neg_ph_q):
                            continue

                    if self.use_ph_sym and f_parity in ((0, 0), (1, 1)):
                        ph_parity_reprs = {}
                        for monomial in f_parity_reprs[f_parity]:
                            ph_parity = monomial.ph_parity()
                            ph_parity_reprs.setdefault(ph_parity, []).append(monomial)
                        for ph_parity in sorted(ph_parity_reprs):
                            self.block_reprs.append(ph_parity_reprs[ph_parity])
                            self.block_momenta.append(q)
                            self.block_keys.append((q, f_parity, ph_parity))
                    else:
                        self.block_reprs.append(f_parity_reprs[f_parity])
                        self.block_momenta.append(q)
                        self.block_keys.append((q, f_parity, None))

    def _build_psd(self):
        r'''
            build momentum PSD blocks M(k) from self._moments_lookup of _build_vars:
            M_{ab}(q) = \sum_r e^{iqr} / N * coeff * m[var_idx] over the list of the pair (a, b), N = Lx*Ly,
            for the lower triangle only
        '''
        self.psd_blocks = []
        n_basis = len(self.basis_reprs)
        n_vars = len(self.vars)
        pair_offsets, pair_sizes, r_terms, var_terms, coeff_terms = self._moments_lookup
        basis_index = {monomial.mask: a for a, monomial in enumerate(self.basis_reprs)}

        for q, block_key, block_basis in zip(self.block_momenta, self.block_keys, self.block_reprs):
            dim = len(block_basis)
            psd = PSDConstraints(
                n_vars=n_vars,
                dim=dim,
                block_key=block_key,
                basis_keys=[self.trans_canon(monomial) for monomial in block_basis],
            )

            phases = np.array([self.fourier_phase(q, (dx, dy)) / self.n_sites for dx in range(self.Lx) for dy in range(self.Ly)])
            # block_basis keeps the order of basis_reprs, so the lower triangle (row >= col) has a >= b
            block_index = np.array([basis_index[monomial.mask] for monomial in block_basis])
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

            # at complex-conjugation invariant momenta (q = -q mod (Lx, Ly)),
            # K maps M(k) to itself and D^H M(k) D is real for D = diag(i^{n_a}),
            # n_a = parity of the number of gamma_2 Majoranas in O_a
            qx, qy = q
            if (2 * qx) % self.Lx == 0 and (2 * qy) % self.Ly == 0:
                psd.realify_phase = [monomial.k_parity(hermitian=False) for monomial in block_basis]
            self.psd_blocks.append(psd)

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        id_expr = self._compile_expr(MajoranaSquareOperator({MajoranaMonomialSquare.identity(self.Lx, self.Ly): 1}))
        if id_expr is None:
            raise ValueError('current basis cannot represent the identity operator')
        id_expr.const = -1
        self.affines.add(id_expr)

        if self.n_particles is not None:
            number_shift = self.number_op.copy()
            number_shift.add(MajoranaMonomialSquare.identity(self.Lx, self.Ly), -float(self.n_particles) / self.n_sites)

            number_expr = self._compile_expr(number_shift)
            if number_expr is None:
                raise ValueError('current basis cannot represent the particle number operator')
            self.affines.add(number_expr)

            number_var_expr = self._compile_expr(number_shift.mul(number_shift))
            if number_var_expr is None:
                raise ValueError('current basis cannot represent the particle number variance operator')
            self.affines.add(number_var_expr)

        # Ward identities <[C, O]> == 0 for each charge C in ward_ops.
        #
        # S^- Ward identities are redundant with S^+:
        # they use the same Majorana bilinear masks and hence generate identical candidates O.
        # because (S^+)^dag = S^- and O^dag = chi_O O, we have
        #
        #   [S^+, O] = -chi_O [S^-, O]^dag.
        #
        # so each S^- Ward identity maps one-to-one to the corresponding S^+ Ward identity
        # after compiled to hermitianized real variables; the same holds for eta^-.
        for charge in self.ward_ops:
            offsets, idxs, coeffs = self._core.build_ward_identities(charge)
            offsets, idxs, coeffs = offsets.tolist(), idxs.tolist(), coeffs.tolist()
            for start, end in zip(offsets[:-1], offsets[1:]):
                self.affines.add(AffineExpr(terms=dict(zip(idxs[start:end], coeffs[start:end])), const=0))
            self.ward_ops[charge] = len(offsets) - 1

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def compile(self, basis_reprs: list[MajoranaMonomialSquare]):
        self.basis_reprs = basis_reprs
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

    def descendants(self, monomial: MajoranaMonomialSquare):
        r'''
            calculate

                [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)

            as a list [(O_\mu(0), (s_x, s_y), coeff), ...]
        '''
        return self._core.descendants(monomial)

    def _get_expr_str(self, expr: AffineExpr) -> str:
        parts = [
            f'{coeff * self.vars[idx].hermitian_phase()}*<{str(self.vars[idx])}>'
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
            'block_momenta': self.block_momenta,
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
            var_cpx=self.var_cpx,
            n_vars=len(self.vars),
            objective=self.obj_expr,
            objective_sense=self.obj_sense,
            psd_blocks=self.psd_blocks,
            affines_mat=self.affines_mat,
            observables=self.obs_exprs,
            energy_ineqs=self.energy_ineqs,
            var_keys=[var.mask for var in self.vars],
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


'''
    basis_reprs helper

    generates translation representatives from local rectangular windows.
    the windows define the locality hierarchy; point-group symmetries are not quotiented here.
'''
def build_basis_reprs(
    Lx: int,
    Ly: int,
    max_degree: int,
    max_support: int,
    windows: list[tuple[int, int]],
    fermion_parity: tuple[int, int] | None = None,
    ph_parity: int | None = None,
):
    n_sites = Lx * Ly
    if max_degree < 0 or max_degree > 4 * n_sites:
        raise ValueError('max_degree must be between 0 and 4*Lx*Ly')
    if max_support < 0 or max_support > n_sites:
        raise ValueError('max_support must be between 0 and Lx*Ly')
    if fermion_parity is not None:
        fermion_parity = tuple(fermion_parity)
        if fermion_parity not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise ValueError('fermion_parity must be one of (0, 0), (0, 1), (1, 0), (1, 1), or None')
    if ph_parity is not None:
        if ph_parity not in (0, 1):
            raise ValueError('ph_parity must be one of 0, 1, or None')
        if fermion_parity not in ((0, 0), (1, 1)):
            raise ValueError('ph_parity filtering is only valid with fermion_parity (0, 0) or (1, 1)')

    parsed_windows = []
    for window in windows:
        if len(window) != 2:
            raise ValueError('each window must be a pair [wx, wy]')
        wx, wy = window
        if wx <= 0 or wx > Lx or wy <= 0 or wy > Ly:
            raise ValueError('window sizes must satisfy 1 <= wx <= Lx and 1 <= wy <= Ly')
        parsed_windows.append((wx, wy))
    parsed_windows = sorted(set(parsed_windows))

    reps = {}
    identity = MajoranaMonomialSquare.identity(Lx, Ly)
    if (
        (fermion_parity is None or identity.fermion_parity(spin=True) == fermion_parity)
        and (ph_parity is None or identity.ph_parity() == ph_parity)
    ):
        reps[identity.trans_canon] = identity.trans_canon_rep

    local_masks = tuple(range(1, 1 << 4))
    for wx, wy in parsed_windows:
        window_sites = tuple(x + Lx * y for y in range(wy) for x in range(wx))
        max_sites = min(max_support, max_degree, len(window_sites))
        for support in range(1, max_sites + 1):
            for sites in itertools.combinations(window_sites, support):
                for masks in itertools.product(local_masks, repeat=support):
                    if sum(mask.bit_count() for mask in masks) > max_degree:
                        continue
                    raw_mask = sum(local_mask << (4 * site) for site, local_mask in zip(sites, masks))
                    cand = MajoranaMonomialSquare(Lx, Ly, raw_mask)
                    if fermion_parity is not None and cand.fermion_parity(spin=True) != fermion_parity:
                        continue
                    if ph_parity is not None and cand.ph_parity() != ph_parity:
                        continue
                    reps.setdefault(cand.trans_canon, cand.trans_canon_rep)

    return [reps[key] for key in sorted(reps)]


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    m1 = MajoranaMonomialSquare.from_str(4, 4, '(0,0)u+ (0,0)u- (2,1)d+')
    m2 = MajoranaMonomialSquare.from_str(4, 4, '(0,0)u+ (3,2)d-')

    print(m1)
    print(m1.mask)
    print(m1.dag_phase())
    print(m1.translate(1, 1))
    print(m1.invert())

    print(*m1.mul(m2))
    print(*m2.mul(m1))
    print(*m1.mul(m1))

    m3 = MajoranaMonomialSquare.from_str(4, 4, '(0,0)u+ (1,0)u+')
    m4 = MajoranaMonomialSquare.from_str(4, 4, '(0,0)d+ (0,0)d-')
    m5 = MajoranaMonomialSquare.from_str(4, 4, '(1,0)u+ (0,1)d+')
    op1 = MajoranaSquareOperator({m3: 1, m4: 2})
    op2 = MajoranaSquareOperator({m5: 3})

    print(op1)
    print(op2)
    print(op1 + op2)
    print(op1.dag())
    print(op1.mul(op2))
    print(op1.commutator(op2))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    params = HubbardSquareParams(Lx=4, Ly=4, t=1.0, U=4.0, n_particles=16)
    compiler = HubbardSquareCompiler(params=params)
    print(compiler.hamil_op)

    # local basis: all Majorana monomials supported on one site
    compiler.compile(basis_reprs=[
        MajoranaMonomialSquare(4, 4, local_mask)
        for local_mask in range(1 << 4)
    ])
    print(compiler.summary())

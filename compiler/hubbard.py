from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from compiler.core import HubbardCompilerCore
from operators.majorana import MajoranaMonomial, MajoranaOperator
from sdp import AffineExpr, PSDConstraints, AffineConstraints, SDPData


@dataclass(slots=True)
class HubbardParams:
    L: int = 8
    t: float = 1.
    U: float = 4.
    # total particle number including both spins; half filling is L
    n_particles: int | None = None


def build_hamil(params: HubbardParams):
    assert params.L >= 2
    hamil_op = MajoranaOperator()

    for x in range(params.L):
        xp1 = (x+1) % params.L
        for spin in ('u', 'd'):
            monomial, sign = MajoranaMonomial.from_str(
                L=params.L, s=f'{x}{spin}+ {xp1}{spin}-', sign=True
            )
            hamil_op.add(monomial, -.5j * params.t * sign / params.L)

            monomial, sign = MajoranaMonomial.from_str(
                L=params.L, s=f'{x}{spin}- {xp1}{spin}+', sign=True
            )
            hamil_op.add(monomial, .5j * params.t * sign / params.L)

        monomial, sign = MajoranaMonomial.from_str(
            L=params.L, s=f'{x}u+ {x}u- {x}d+ {x}d-', sign=True
        )
        hamil_op.add(monomial, -.25 * params.U * sign / params.L)

    return hamil_op


def build_number(params: HubbardParams, spin: str | None = None):
    assert spin in (None, 'u', 'd')
    spins = ('u', 'd') if spin is None else (spin,)
    number_op = MajoranaOperator()
    for s in spins:
        number_op.add(MajoranaMonomial.identity(params.L), .5)
        for x in range(params.L):
            monomial, sign = MajoranaMonomial.from_str(
                L=params.L, s=f'{x}{s}+ {x}{s}-', sign=True
            )
            number_op.add(monomial, .5j * sign / params.L)
    return number_op


def build_sz(params: HubbardParams, x: int):
    assert 0 <= x < params.L
    sz_op = MajoranaOperator()
    for spin, coeff in (('u', .25j), ('d', -.25j)):
        monomial, sign = MajoranaMonomial.from_str(
            L=params.L, s=f'{x}{spin}+ {x}{spin}-', sign=True
        )
        sz_op.add(monomial, coeff * sign)
    return sz_op


def build_szz(params: HubbardParams, r: int):
    assert 0 <= r <= params.L//2
    return build_sz(params, 0).mul(build_sz(params, r))


def build_double_occ(params: HubbardParams):
    double_occ_op = MajoranaOperator({MajoranaMonomial.identity(params.L): .25})
    for x in range(params.L):
        for spin in ('u', 'd'):
            monomial, sign = MajoranaMonomial.from_str(
                L=params.L, s=f'{x}{spin}+ {x}{spin}-', sign=True
            )
            double_occ_op.add(monomial, .25j * sign / params.L)

        monomial, sign = MajoranaMonomial.from_str(
            L=params.L, s=f'{x}u+ {x}u- {x}d+ {x}d-', sign=True
        )
        double_occ_op.add(monomial, -.25 * sign / params.L)
    return double_occ_op


class HubbardCompiler:
    L: int
    params: HubbardParams

    basis_reprs: list[MajoranaMonomial]
    block_reprs: list[list[MajoranaMonomial]]
    block_momenta: list[int]
    block_keys: list[tuple[int, tuple[int, int], int | None]]

    _core: HubbardCompilerCore

    # SDP variables are real expectations of hermitianized Majorana monomials
    # with even fermion parity for each spin, even K parity, and even particle-hole parity at half filling
    var_cpx: bool = False
    vars: list[MajoranaMonomial]
    _moments_lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None

    ward_ops: dict[str, int]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaOperator
    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    # SDP objective
    obj_op: MajoranaOperator
    obj_expr: AffineExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[AffineExpr]

    # uncertified observables
    obs_ops: dict[str, MajoranaOperator | list[MajoranaOperator]]
    obs_exprs: dict[str, AffineExpr | list[AffineExpr]]

    def __init__(self,
        params: HubbardParams,
        *,
        obj_op: MajoranaOperator | None = None,
        obj_sense: str = 'min',
        e_lb: float = None,
        e_ub: float = None,
        obs_ops: dict | None = None,
    ):
        if params.L <= 0:
            raise ValueError('L must be positive')
        if params.n_particles is not None and not (0 <= params.n_particles <= 2 * params.L):
            raise ValueError('n_particles must be between 0 and 2L')

        self.params = params
        self.L = params.L
        self.n_particles = params.n_particles

        self.use_ph_sym = self.L % 2 == 0 and self.n_particles in (None, self.L)
        self.use_eta_sym = self.use_ph_sym
        self.ward_ops = {'hamil': 0, 'Nu': 0, 'Nd': 0, 'S+': 0}
        if self.use_eta_sym:
            self.ward_ops['eta+'] = 0

        self.hamil_op = build_hamil(params)
        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op

        # the C++ core takes over:
        #   1) building the SDP variables and the moments lookup (OpenMP),
        #   2) generating the Ward identities (OpenMP),
        #   3) generating the descendants,
        #   4) compiling Majorana operators into linear expressions of the SDP variables.
        # the Hamiltonian coefficients are scaled from energy density to total H.
        self._core = HubbardCompilerCore(
            L=self.L,
            hamil_monomials=list(self.hamil_op.terms),
            hamil_coeffs=[self.L * h_coeff for h_coeff in self.hamil_op.terms.values()],
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

    def trans_canon(self, monomial: MajoranaMonomial) -> int:
        return monomial.trans_canon

    def trans_canon_rep(self, monomial: MajoranaMonomial) -> MajoranaMonomial:
        return monomial.trans_canon_rep

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
    def nonzero_fourier(monomial: MajoranaMonomial, q: int) -> bool:
        if monomial.period_sign == 1:
            return (q * monomial.period) % monomial.L == 0
        return (2 * q * monomial.period) % (2 * monomial.L) == monomial.L

    def _build_block_reprs(self):
        '''
            for a Majorana monomial with period L_a < L,
            allowed momentum satisfy e^{-i k L_a} = s where s is the period_sign
        '''
        self.block_reprs = []
        self.block_momenta = []
        self.block_keys = [] # symmetry sector (q, fermion parity, particle-hole parity or None)
        for q in range(self.L//2 + 1):
            f_parity_reprs = {}
            for monomial in self.basis_reprs:
                if not self.nonzero_fourier(monomial, q):
                    continue
                f_parity = monomial.fermion_parity(spin=True)
                f_parity_reprs.setdefault(f_parity, []).append(monomial)

            for f_parity in sorted(f_parity_reprs):
                if self.use_ph_sym and f_parity in ((0, 1), (1, 0)) and q > self.L//4:
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
            M_{ab}(q) = \sum_r e^{iqr} / L * coeff * m[var_idx] over the list of the pair (a, b),
            for the lower triangle only
        '''
        self.psd_blocks = []
        n_basis = len(self.basis_reprs)
        n_vars = len(self.vars)
        pair_offsets, pair_sizes, r_terms, var_terms, coeff_terms = self._moments_lookup
        basis_index = {monomial.mask: a for a, monomial in enumerate(self.basis_reprs)}

        # K symmetry equals M(k) \succcurlyeq 0 and M(-k) \succcurlyeq 0
        for q, block_key, block_basis in zip(self.block_momenta, self.block_keys, self.block_reprs):
            dim = len(block_basis)
            psd = PSDConstraints(
                n_vars=n_vars,
                dim=dim,
                block_key=block_key,
                basis_keys=[self.trans_canon(monomial) for monomial in block_basis],
            )

            phases = np.array([self.fourier_phase(q, r) / self.L for r in range(self.L)])
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

            # at complex-conjugation invariant momenta (q = -q mod L),
            # K maps M(k) to itself and D^H M(k) D is real for D = diag(i^{n_a}),
            # n_a = parity of the number of gamma_2 Majoranas in O_a
            if (2 * q) % self.L == 0:
                psd.realify_phase = np.array(
                    [monomial.k_parity(hermitian=False) for monomial in block_basis], dtype=np.int64
                )
            self.psd_blocks.append(psd)

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        id_expr = self._compile_expr(MajoranaOperator({MajoranaMonomial.identity(self.L): 1}))
        if id_expr is None:
            raise ValueError('current basis cannot represent the identity operator')
        id_expr.const = -1
        self.affines.add(id_expr)

        # fix total particle number
        if self.n_particles is not None:
            number_shift = self.number_op.copy()
            number_shift.add(MajoranaMonomial.identity(self.L), -float(self.n_particles) / self.L)

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
            self.affines.add_batch(offsets, idxs, coeffs)
            self.ward_ops[charge] = len(offsets) - 1

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _compile_expr(self, op: MajoranaOperator) -> AffineExpr | None:
        # None if a term cannot be represented by an SDP variable
        expr = self._core.compile_expr(list(op.terms), list(op.terms.values()))
        if expr is None:
            return None
        idxs, coeffs = expr
        return AffineExpr(terms=dict(zip(idxs.tolist(), coeffs.tolist())), const=0)

    def compile(self, basis_reprs: list[MajoranaMonomial]):
        # build SDP variables
        self.basis_reprs = basis_reprs
        self._build_vars()

        # build SDP objective
        self.obj_expr = self._compile_expr(self.obj_op)
        if self.obj_expr is None:
            raise ValueError('current basis cannot represent the SDP objective')

        # sandwich the energy for certified observables
        self.energy_ineqs = []
        hamil_expr = self._compile_expr(self.hamil_op)
        if hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        if self.e_lb is not None:
            self.energy_ineqs.append(AffineExpr(
                terms=hamil_expr.terms.copy(),
                const=hamil_expr.const - self.e_lb
            ))
        if self.e_ub is not None:
            self.energy_ineqs.append(AffineExpr(
                terms={idx: -coeff for idx, coeff in hamil_expr.terms.items()},
                const=self.e_ub - hamil_expr.const
            ))

        # build observables
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

        # build PSD blocks
        self._build_block_reprs()
        self._build_psd()
        self._moments_lookup = None

        # build affine constraints
        self._build_affines()

    def descendants(self, monomial: MajoranaMonomial):
        r'''
            calculate

                [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)

            as a list [(O_\mu(0), s, coeff), ...]
        '''
        return self._core.descendants(monomial)

    def _get_expr_str(self, expr: AffineExpr) -> str:
        parts = [
            f'{coeff*self.vars[idx].hermitian_phase()}*<{str(self.vars[idx])}>'
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


'''
    basis_reprs helper

    for each majorana monomial,
        1) max_degree restricts the max number of majorana operator;
        2) max_support restricts the max number of lattice sites involved;
        3) max_diameter restricts the max distance of any two majorana operators;
        4) fermion_parity, when provided, keeps only monomials with the given
           spin-resolved fermion parity (N_up mod 2, N_down mod 2);
        5) ph_parity, when provided, keeps only monomials with the given particle-hole parity;
           it is only allowed in the 00/11 fermion parity sectors because particle-hole
           symmetry does not commute with single-site translation in the 01/10 sectors.
'''
def build_basis_reprs(
    L: int,
    max_degree: int,
    max_support: int,
    max_diameter: int,
    fermion_parity: tuple[int, int] | None = None,
    ph_parity: int | None = None,
):
    if max_degree < 0 or max_degree > 4 * L:
        raise ValueError('max_degree must be between 0 and 4L')
    if max_support < 0 or max_support > L:
        raise ValueError('max_support must be between 0 and L')
    if max_diameter < 0 or max_diameter > L // 2:
        raise ValueError('max_diameter must be between 0 and ⌊L/2⌋')
    if fermion_parity is not None:
        fermion_parity = tuple(fermion_parity)
        if fermion_parity not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise ValueError('fermion_parity must be one of (0, 0), (0, 1), (1, 0), (1, 1), or None')
    if ph_parity is not None:
        if ph_parity not in (0, 1):
            raise ValueError('ph_parity must be one of 0, 1, or None')
        if fermion_parity not in ((0, 0), (1, 1)):
            raise ValueError('ph_parity filtering is only valid with fermion_parity (0, 0) or (1, 1)')

    reps = {}
    identity = MajoranaMonomial.identity(L=L)
    if (
        (fermion_parity is None or identity.fermion_parity(spin=True) == fermion_parity)
        and (ph_parity is None or identity.ph_parity() == ph_parity)
    ):
        reps[identity.trans_canon] = identity.trans_canon_rep

    local_masks = tuple(range(1, 1 << 4))
    max_sites = min(max_support, max_degree, L)

    for support in range(1, max_sites + 1):
        for sites_tail in itertools.combinations(range(1, L), support - 1):
            sites = (0, *sites_tail)
            diameter = 0
            for i, site1 in enumerate(sites):
                for site2 in sites[i+1:]:
                    dist = abs(site1 - site2)
                    diameter = max(diameter, min(dist, L - dist))
            if diameter > max_diameter:
                continue

            for masks in itertools.product(local_masks, repeat=support):
                if sum(mask.bit_count() for mask in masks) > max_degree:
                    continue
                mask = sum(
                    local_mask << (4 * site)
                    for site, local_mask in zip(sites, masks)
                )
                cand = MajoranaMonomial(L=L, mask=mask)
                if fermion_parity is not None and cand.fermion_parity(spin=True) != fermion_parity:
                    continue
                if ph_parity is not None and cand.ph_parity() != ph_parity:
                    continue
                reps.setdefault(cand.trans_canon, cand.trans_canon_rep)

    return [reps[key] for key in sorted(reps)]


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    m1 = MajoranaMonomial.from_str(L=8, s='0u+ 0u- 2d+')
    m2 = MajoranaMonomial.from_str(L=8, s='0u+ 3d-')

    print(m1)
    print(m1.mask)
    print(m1.dag_phase())
    print(m1.translate(1))
    print(m1.invert())

    print(*m1.mul(m2))
    print(*m2.mul(m1))
    print(*m1.mul(m1))

    m3 = MajoranaMonomial.from_str(L=8, s='0u+ 1u+')
    m4 = MajoranaMonomial.from_str(L=8, s='0d+ 0d-')
    m5 = MajoranaMonomial.from_str(L=8, s='1u+ 2d+')
    op1 = MajoranaOperator({m3: 1, m4: 2})
    op2 = MajoranaOperator({m5: 3})

    print(op1)
    print(op2)
    print(op1 + op2)
    print(op1.dag())
    print(op1.mul(op2))
    print(op1.commutator(op2))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    params = HubbardParams(L=8, t=1., U=4., n_particles=8)
    compiler = HubbardCompiler(params=params)
    print(compiler.hamil_op)

    # local basis: all majorana monomials supported on one site
    compiler.compile(basis_reprs=build_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    ))
    print(compiler.summary())

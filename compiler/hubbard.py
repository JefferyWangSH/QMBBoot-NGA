from dataclasses import dataclass
from lru import LRU
import itertools
import os
import numpy as np
import scipy as sp

from operators.majorana import MajoranaMonomial, MajoranaOperator, _SPIN_LADDER_TERMS
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData

_USE_JIT = os.environ.get('USE_JIT', '1') != '0'
if _USE_JIT:
    try:
        from compiler.kernels.hubbard import sym_canon as _cpp_sym_canon
    except ImportError:
        _cpp_sym_canon = None
else:
    _cpp_sym_canon = None

_CACHE_MAX_SIZE = 10_000_000


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


class HubbardCompiler:
    L: int
    params: HubbardParams

    basis_reprs: list[MajoranaMonomial]
    block_reprs: list[list[MajoranaMonomial]]
    block_momenta: list[int]

    # SDP variables are real expectations of
    # hermitianized Majorana monomials with even fermion parity for each spin and even K parity
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[MajoranaMonomial]

    ward_ops: dict[str, int]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaOperator
    _hamil_terms_at_site: list[list[tuple[int, float | complex, tuple[int, ...]]]]

    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    # SDP objective
    obj_op: MajoranaOperator
    obj_expr: LinearExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[LinearExpr]

    # uncertified observables
    obs_ops: dict[str, MajoranaOperator | list[MajoranaOperator]]
    obs_exprs: dict[str, LinearExpr | list[LinearExpr]]

    def __init__(self,
        params: HubbardParams,
        *,
        obj_op: MajoranaOperator | None = None,
        obj_sense: str = 'min',
        e_lb: float = None,
        e_ub: float = None,
        obs_ops: dict | None = None,
    ):
        self.L = params.L
        self.params = params
        self.n_particles = params.n_particles
        self.ward_ops = {'hamil': 0, 'Nu': 0, 'Nd': 0, 'S+': 0}

        self.hamil_op = build_hamil(params)

        # site index for local Hamiltonian terms
        # each entry stores (h_mask, h_coeff, lower_masks):
        # 1) h_mask is the Majorana mask of a local H term touching this site;
        # 2) h_coeff is scaled from energy density to total H;
        # 3) and lower_masks cache swap signs.
        self._hamil_terms_at_site = [[] for _ in range(self.L)]
        for h_mono, h_coeff in self.hamil_op.terms.items():
            h_mask = h_mono.mask
            support = h_mask
            lower_masks = []
            while support:
                bit = support & -support
                mode = bit.bit_length() - 1
                lower_masks.append((1 << mode) - 1)
                support ^= bit

            support = h_mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 4
                support &= ~(0xf << (4*site))
                self._hamil_terms_at_site[site].append((h_mask, self.L * h_coeff, tuple(lower_masks)))

        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op

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
                raise ValueError('Hamiltonian objective only supports min')
            if e_lb is not None or e_ub is not None:
                raise ValueError('Hamiltonian objective does not use energy bounds e_lb or e_ub')
            self.e_lb = None
            self.e_ub = None

        self.obs_ops = {} if obs_ops is None else obs_ops.copy()

    def trans_canon(self, monomial: MajoranaMonomial) -> int:
        if not hasattr(self, '_trans_canon_cache'):
            self._trans_canon_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._trans_canon_cache:
            return self._trans_canon_cache[monomial.mask]

        key = MajoranaMonomial(self.L, monomial.mask).trans_canon
        self._trans_canon_cache[monomial.mask] = key
        return key

    def trans_canon_rep(self, monomial: MajoranaMonomial) -> MajoranaMonomial:
        return MajoranaMonomial(self.L, self.trans_canon(monomial))

    def _sym_canon(self, monomial: MajoranaMonomial, sign: bool = False) -> int | tuple[int, int] | None:
        if not hasattr(self, '_sym_canon_cache'):
            self._sym_canon_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._sym_canon_cache:
            canon = self._sym_canon_cache[monomial.mask]
            if canon is None:
                return None
            if sign:
                return canon
            return canon[0]

        if _cpp_sym_canon is not None:
            canon = _cpp_sym_canon(monomial, sign=True)
            self._sym_canon_cache[monomial.mask] = canon
            if canon is None:
                return None
            if sign:
                return canon
            return canon[0]

        if monomial.period_sign == -1:
            self._sym_canon_cache[monomial.mask] = None
            return None

        cands = []
        for up_quarters, dn_quarters in itertools.product(range(4), repeat=2):
            rotated, rot_sign = monomial.c4_rotate(up_quarters, dn_quarters)

            for exchange in (False, True):
                exchanged = rotated
                exchange_sign = rot_sign
                if exchange:
                    exchanged, step_sign = rotated.spin_exchange()
                    exchange_sign *= step_sign

                for invert in (False, True):
                    cand = exchanged
                    cand_sign = exchange_sign
                    if invert:
                        cand, step_sign = exchanged.invert()
                        cand_sign *= step_sign

                    cands.append((
                        cand.trans_canon,
                        cand_sign * cand.trans_canon_sign,
                    ))

        canon_key = min(cand_key for cand_key, _ in cands)
        signs = {cand_sign for cand_key, cand_sign in cands if cand_key == canon_key}
        if len(signs) > 1:
            self._sym_canon_cache[monomial.mask] = None
            return None

        canon_sign = signs.pop()
        self._sym_canon_cache[monomial.mask] = (canon_key, canon_sign)
        if sign:
            return canon_key, canon_sign
        return canon_key

    def _sym_allowed(self, monomial: MajoranaMonomial) -> bool:
        '''
            SDP variables should have even fermion parity for each spin and even K parity.
        '''
        if not hasattr(self, '_sym_allowed_cache'):
            self._sym_allowed_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._sym_allowed_cache:
            return self._sym_allowed_cache[monomial.mask]

        allowed = (
            monomial.fermion_parity(spin=True) == (0, 0)
            and monomial.k_parity(hermitian=True) == 0
        )
        self._sym_allowed_cache[monomial.mask] = allowed
        return allowed

    @staticmethod
    def _moment(monomial1: MajoranaMonomial, monomial2: MajoranaMonomial) -> tuple[MajoranaMonomial, complex]:
        r'''
            O_1^\dag O_2 with O_1, O_2 Majorana monomials
        '''
        monomial, sign = monomial1.mul(monomial2)
        return monomial, sign * monomial1.dag_phase()

    def _build_vars(self):
        '''
            build SDP variables
        '''
        self.var_index = {}
        self.vars = []

        for monomial1 in self.basis_reprs:
            for r in range(self.L):
                monomial1r, _ = monomial1.translate(r)
                for monomial2 in self.basis_reprs:
                    monomial, _ = self._moment(monomial1r, monomial2)

                    if self._sym_allowed(monomial):
                        key = self._sym_canon(monomial)
                        if key is None:
                            continue
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(MajoranaMonomial(L=self.L, mask=key))

    @staticmethod
    def nonzero_fourier(monomial: MajoranaMonomial, n: int) -> bool:
        if monomial.period_sign == 1:
            return (n * monomial.period) % monomial.L == 0
        return (2 * n * monomial.period) % (2 * monomial.L) == monomial.L

    def _build_block_reprs(self):
        '''
            for a Majorana monomial with period L_a < L,
            allowed momentum satisfy e^{-i k L_a} = s where s is the period_sign
        '''
        self.block_reprs = []
        self.block_momenta = []
        for n in range(self.L//2 + 1):
            parity_reprs = {}
            for monomial in self.basis_reprs:
                if not self.nonzero_fourier(monomial, n):
                    continue
                parity = monomial.fermion_parity(spin=True)
                parity_reprs.setdefault(parity, []).append(monomial)

            for parity in sorted(parity_reprs):
                self.block_reprs.append(parity_reprs[parity])
                self.block_momenta.append(n)

    def _build_psd(self):
        '''
            build momentum PSD blocks M(k)
        '''
        self.psd_blocks = []

        # K symmetry equals M(k) \succcurlyeq 0 and M(-k) \succcurlyeq 0
        for n, block_basis in zip(self.block_momenta, self.block_reprs):
            k = 2*np.pi * n / self.L
            psd = PSDConstraints(n_vars=len(self.vars), dim=len(block_basis))

            for row, monomial1 in enumerate(block_basis):
                for col, monomial2 in enumerate(block_basis):
                    expr = {}
                    for r in range(self.L):
                        monomial1r, rot_sign = monomial1.translate(r)
                        monomial, mul_sign = self._moment(monomial1r, monomial2)

                        if not self._sym_allowed(monomial):
                            continue
                        canon = self._sym_canon(monomial, sign=True)
                        if canon is None:
                            continue
                        key, canon_sign = canon

                        # combine Fourier phase, translation/multiplication/canonical signs,
                        # then divide by the hermitian phase so moment variables stay real.
                        coeff = (
                            np.exp(1j * k * r) / self.L
                            * rot_sign * mul_sign * canon_sign
                            / monomial.hermitian_phase()
                        )
                        idx = self.var_index[key]
                        expr[idx] = expr.get(idx, 0) + coeff
                        if abs(expr[idx]) < 1e-14:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))
            self.psd_blocks.append(psd)

    def _hamil_comm(self, monomial: MajoranaMonomial) -> MajoranaOperator:
        r'''
            return [\sum_i H_i, O] as a Majorana operator
        '''
        op = MajoranaOperator()
        seen = set()
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4*site))

            for h_mask, h_coeff, lower_masks in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                anticomm = (h_mask & monomial.mask).bit_count() & 1
                if anticomm:
                    sign = -1 if sum((monomial.mask & lower_mask).bit_count() for lower_mask in lower_masks) & 1 else 1
                    prod = MajoranaMonomial(self.L, h_mask ^ monomial.mask)
                    op.add(prod, 2 * h_coeff * sign)
        return op

    def _add_hamil_wards(self):
        r'''
            add representable stationarity Ward identities <[H, O]> == 0
            within the current SDP variable set

            candidate O must have odd K parity and even spin-resolved fermion parity.
        '''
        seen_masks = set()
        seen_keys = set()

        for var in self.vars:
            support = var.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 4
                support &= ~(0xf << (4*site))

                for h_mask, _, _ in self._hamil_terms_at_site[site]:
                    anticomm = (h_mask & var.mask).bit_count() & 1
                    if not anticomm:
                        continue

                    mask = var.mask ^ h_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = MajoranaMonomial(self.L, mask)
                    # trans_canon is cheaper than _sym_canon
                    # although it may produce redundant Ward identities
                    key = self.trans_canon(cand)
                    # key = self._sym_canon(cand)
                    # if key is None:
                    #     continue
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_expr(self._hamil_comm(cand))
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops['hamil'] += 1

    def _number_comm(self, monomial: MajoranaMonomial, spin: str) -> MajoranaOperator:
        r'''
            return [\sum_i N_{i,s}, O] as a Majorana operator
            spin: 'u' or 'd'
        '''
        assert spin in ('u', 'd')
        spin_offset = 0 if spin == 'u' else 2
        op = MajoranaOperator()

        support = monomial.mask
        while support:
            bit = support & -support
            mode = bit.bit_length() - 1
            site, rem = divmod(mode, 4)
            support ^= bit
            if rem // 2 != spin_offset // 2:
                continue

            lo_bit = 1 << (4*site + spin_offset)
            number_mask = lo_bit | (lo_bit << 1)
            overlap = monomial.mask & number_mask
            if overlap == 0 or overlap == number_mask:
                continue

            prod = MajoranaMonomial(self.L, monomial.mask ^ number_mask)
            sign = -1 if overlap == lo_bit else 1
            op.add(prod, 1j * sign)
        return op

    def _add_number_wards(self, spin: str):
        r'''
            add representable U(1) Ward identities <[N_s, O]> == 0
            within the current SDP variable set

            candidate O must have odd K parity and even spin-resolved fermion parity.
        '''
        assert spin in ('u', 'd')
        spin_offset = 0 if spin == 'u' else 2
        seen_masks = set()
        seen_keys = set()

        for var in self.vars:
            support = var.mask
            while support:
                bit = support & -support
                mode = bit.bit_length() - 1
                site, rem = divmod(mode, 4)
                support ^= bit
                # bypass mismatched spin sector
                if rem // 2 != spin_offset // 2:
                    continue

                number_mask = 0b11 << (4*site + spin_offset)
                if (var.mask & number_mask).bit_count() != 1:
                    continue
                mask = var.mask ^ number_mask
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)

                cand = MajoranaMonomial(self.L, mask)
                # trans_canon is cheaper than _sym_canon
                # although it may produce redundant Ward identities
                key = self.trans_canon(cand)
                # key = self._sym_canon(cand)
                # if key is None:
                #     continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                expr = self._compile_expr(self._number_comm(cand, spin))
                if expr is None or not expr.terms:
                    continue
                self.affines.add(expr)
                self.ward_ops[f'N{spin}'] += 1

    def _spin_comm(self, monomial: MajoranaMonomial, spin: str) -> MajoranaOperator:
        r'''
            return [\sum_i S_{i,s}, O] as a Majorana operator
            spin: '+' or '-'
        '''
        assert spin in ('+', '-')
        op = MajoranaOperator()
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4*site))

            for rem1, rem2, coeff in _SPIN_LADDER_TERMS[spin]:
                mode1 = 4*site + rem1
                mode2 = 4*site + rem2
                spin_mask = (1 << mode1) | (1 << mode2)
                if (monomial.mask & spin_mask).bit_count() != 1:
                    continue

                sign = -1 if (
                    (monomial.mask & ((1 << mode1) - 1)).bit_count()
                    + (monomial.mask & ((1 << mode2) - 1)).bit_count()
                ) & 1 else 1
                prod = MajoranaMonomial(self.L, monomial.mask ^ spin_mask)
                op.add(prod, 2 * coeff * sign)
        return op

    def _add_spin_wards(self, spin: str):
        r'''
            add representable SU(2) Ward identities <[S_\pm, O]> == 0
            within the current SDP variable set

                1) <[S_z, O]> == 0 have been covered by <[N_s, O]> == 0.

                2) since S_\pm carries spin-resolved parity (1, 1), candidate O has
                   spin-resolved parity (1, 1) so that [S_\pm, O] can land in the even
                   variable sector (0, 0).

                3) its K parity is not fixed because S_\pm mixes local K parities.
        '''
        assert spin in ('+', '-')
        seen_masks = set()
        seen_keys = set()

        for var in self.vars:
            support = var.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 4
                support &= ~(0xf << (4*site))

                for rem1, rem2, _ in _SPIN_LADDER_TERMS[spin]:
                    spin_mask = (1 << (4*site + rem1)) | (1 << (4*site + rem2))
                    if (var.mask & spin_mask).bit_count() != 1:
                        continue
                    mask = var.mask ^ spin_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = MajoranaMonomial(self.L, mask)
                    # trans_canon is cheaper than _sym_canon
                    # although it may produce redundant Ward identities
                    key = self.trans_canon(cand)
                    # key = self._sym_canon(cand)
                    # if key is None:
                    #     continue
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_expr(self._spin_comm(cand, spin))
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops[f'S{spin}'] += 1

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))
        for key in self.ward_ops:
            self.ward_ops[key] = 0

        id_key = self._sym_canon(MajoranaMonomial.identity(self.L))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

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

        # Ward identities
        self._add_hamil_wards()
        self._add_number_wards('u')
        self._add_number_wards('d')

        # S^- Ward identities are redundant with S^+:
        # they use the same Majorana bilinear masks and hence generate identical candidates O.
        # because (S^+)^dag = S^-, we have
        #
        #   [S^+, O] = -eta_O [S^-, O]^dag
        #
        # for O^dag = eta_O O. After compiling to hermitianized real variables,
        # each S^- Ward identity maps one-to-one to the corresponding S^+ Ward identity.
        self._add_spin_wards('+')
        # self._add_spin_wards('-')

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _compile_expr(self, op: MajoranaOperator) -> LinearExpr | None:
        expr = {}
        for monomial, coeff in op.terms.items():
            if not self._sym_allowed(monomial):
                continue
            canon = self._sym_canon(monomial, sign=True)
            if canon is None:
                continue
            key, sign = canon
            if key not in self.var_index:
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff * sign / monomial.hermitian_phase()
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

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
            self.energy_ineqs.append(LinearExpr(
                terms=hamil_expr.terms.copy(),
                const=hamil_expr.const - self.e_lb
            ))
        if self.e_ub is not None:
            self.energy_ineqs.append(LinearExpr(
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

        # build affine constraints
        self._build_affines()

    def descendants(self, monomial: MajoranaMonomial):
        r'''
            calculate

                C_a = [H, O_a(0)] = \sum_{b,s} C_{ab}(s) T_s O'(0)_b

            as entry list [(O'(0)_b, s, C_{ab}(s)), ...]
        '''
        if not hasattr(self, '_descendants_cache'):
            self._descendants_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._descendants_cache:
            return self._descendants_cache[monomial.mask]

        entries = {}
        comm_op = self._hamil_comm(monomial)

        for desc, coeff in comm_op.terms.items():
            desc_rep = self.trans_canon_rep(desc)
            s, s_sign = 0, 1
            for shift in range(desc.L):
                shifted, sign = desc_rep.translate(shift)
                if shifted.mask == desc.mask:
                    s, s_sign = shift, sign
                    break

            key = (desc_rep.mask, s)
            entries[key] = entries.get(key, 0) + coeff * s_sign
            if entries[key] == 0:
                del entries[key]

        descs = [
            (MajoranaMonomial(L=self.L, mask=desc_mask), s, coeff)
            for (desc_mask, s), coeff in entries.items()
        ]
        self._descendants_cache[monomial.mask] = descs
        return descs

    def _get_expr_str(self, expr: LinearExpr) -> str:
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
        3) max_diameter restricts the max distance of any two majorana operators.
'''
def build_basis_reprs(L: int, max_degree: int, max_support: int, max_diameter: int):
    if max_degree < 0 or max_degree > 4 * L:
        raise ValueError('max_degree must be between 0 and 4L')
    if max_support < 0 or max_support > L:
        raise ValueError('max_support must be between 0 and L')
    if max_diameter < 0 or max_diameter > L // 2:
        raise ValueError('max_diameter must be between 0 and L//2')

    def site_diameter(sites: tuple[int, ...]) -> int:
        diam = 0
        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                dist = abs(site1 - site2)
                diam = max(diam, min(dist, L-dist))
        return diam

    def shifted_mask(sites: tuple[int, ...], masks: tuple[int, ...], anchor: int) -> int:
        shift = sites[anchor]
        return sum(
            local_mask << (4 * ((site - shift) % L))
            for site, local_mask in zip(sites, masks)
        )

    reps = [MajoranaMonomial.identity(L=L)]
    local_masks = tuple(range(1, 1 << 4))
    max_sites = min(max_support, max_degree, L)

    for support in range(1, max_sites + 1):
        for sites_tail in itertools.combinations(range(1, L), support - 1):
            sites = (0, *sites_tail)
            if site_diameter(sites) > max_diameter:
                continue

            for masks in itertools.product(local_masks, repeat=support):
                if sum(mask.bit_count() for mask in masks) > max_degree:
                    continue
                mask = shifted_mask(sites, masks, anchor=0)
                if mask != min(shifted_mask(sites, masks, anchor) for anchor in range(support)):
                    continue
                reps.append(MajoranaMonomial(L=L, mask=mask))
    return reps


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

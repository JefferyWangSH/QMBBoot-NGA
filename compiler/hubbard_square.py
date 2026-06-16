from dataclasses import dataclass
from lru import LRU
import itertools
import os
import numpy as np
import scipy as sp

from operators.majorana import _SPIN_LADDER_TERMS
from operators.majorana_square import MajoranaMonomialSquare, MajoranaSquareOperator
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData

_USE_JIT = os.environ.get('USE_JIT', '1') != '0'
if _USE_JIT:
    try:
        from compiler.kernels.hubbard_square import sym_canon as _cpp_sym_canon
    except ImportError:
        _cpp_sym_canon = None
else:
    _cpp_sym_canon = None

_CACHE_MAX_SIZE = 10_000_000

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


class HubbardSquareCompiler:
    params: HubbardSquareParams
    Lx: int
    Ly: int
    n_sites: int
    n_particles: int | None

    basis_reprs: list[MajoranaMonomialSquare]
    block_reprs: list[list[MajoranaMonomialSquare]]
    block_momenta: list[tuple[int, int]]

    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[MajoranaMonomialSquare]

    ward_ops: dict[str, int]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaSquareOperator | None
    _hamil_terms_at_site: list[list[tuple[int, float | complex, tuple[int, ...]]]]
    number_op: MajoranaSquareOperator | None

    obj_op: MajoranaSquareOperator | None
    obj_expr: LinearExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[LinearExpr]

    obs_ops: dict[str, MajoranaSquareOperator | list[MajoranaSquareOperator]]
    obs_exprs: dict[str, LinearExpr | list[LinearExpr]]

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
        self.ward_ops = {'hamil': 0, 'Nu': 0, 'Nd': 0, 'S+': 0}

        self.hamil_op = build_hamil(params)
        self._hamil_terms_at_site = [[] for _ in range(self.n_sites)]
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
                support &= ~(0xf << (4 * site))
                self._hamil_terms_at_site[site].append((h_mask, self.n_sites * h_coeff, tuple(lower_masks)))

        self.number_op = build_number(params)

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

    def trans_canon(self, monomial: MajoranaMonomialSquare) -> int:
        if not hasattr(self, '_trans_canon_cache'):
            self._trans_canon_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._trans_canon_cache:
            return self._trans_canon_cache[monomial.mask]

        key = MajoranaMonomialSquare(self.Lx, self.Ly, monomial.mask).trans_canon
        self._trans_canon_cache[monomial.mask] = key
        return key

    def trans_canon_rep(self, monomial: MajoranaMonomialSquare) -> MajoranaMonomialSquare:
        return MajoranaMonomialSquare(self.Lx, self.Ly, self.trans_canon(monomial))

    def fourier_phase(self, q: tuple[int, int], r: tuple[int, int]) -> complex:
        # q is the integer momentum-sector index
        qx, qy = q
        x, y = r
        return np.exp(1j * 2 * np.pi * (qx * x / self.Lx + qy * y / self.Ly))

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

    def _sym_allowed(self, monomial: MajoranaMonomialSquare) -> bool:
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

    def _sym_canon(self, monomial: MajoranaMonomialSquare, sign: bool = False):
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

        cands = []
        for up_quarters, dn_quarters in itertools.product(range(4), repeat=2):
            majorana_rotated, majorana_rot_sign = monomial.majorana_c4_rotate(up_quarters, dn_quarters)

            for exchange in (False, True):
                exchanged = majorana_rotated
                exchange_sign = majorana_rot_sign
                if exchange:
                    exchanged, step_sign = majorana_rotated.spin_exchange()
                    exchange_sign *= step_sign

                lattice_quarters = range(4) if self.Lx == self.Ly else (0, 2)
                for quarters in lattice_quarters:
                    lattice_rotated, lattice_rot_sign = exchanged.lattice_c4_rotate(quarters)
                    lattice_rot_sign *= exchange_sign

                    for reflect in (False, True):
                        cand = lattice_rotated
                        cand_sign = lattice_rot_sign
                        if reflect:
                            cand, step_sign = lattice_rotated.lattice_reflect_x()
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

    @staticmethod
    def _moment(monomial1: MajoranaMonomialSquare, monomial2: MajoranaMonomialSquare):
        monomial, sign = monomial1.mul(monomial2)
        return monomial, sign * monomial1.dag_phase()

    def _compile_expr(self, op: MajoranaSquareOperator) -> LinearExpr | None:
        expr = {}
        for monomial, coeff in op.terms.items():
            if not self._sym_allowed(monomial):
                continue
            canon = self._sym_canon(monomial, sign=True)
            if canon is None:
                continue
            key, canon_sign = canon
            if key not in self.var_index:
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff * canon_sign / monomial.hermitian_phase()
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def _build_vars(self):
        self.var_index = {}
        self.vars = []

        for monomial1 in self.basis_reprs:
            for dx in range(self.Lx):
                for dy in range(self.Ly):
                    monomial1r, _ = monomial1.translate(dx, dy)
                    for monomial2 in self.basis_reprs:
                        monomial, _ = self._moment(monomial1r, monomial2)

                        if not self._sym_allowed(monomial):
                            continue
                        key = self._sym_canon(monomial)
                        if key is None:
                            continue
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(MajoranaMonomialSquare(self.Lx, self.Ly, key))

    def _build_block_reprs(self):
        self.block_reprs = []
        self.block_momenta = []
        for nx in range(self.Lx):
            for ny in range(self.Ly):
                # complex-conjugation K symmetry makes k and -k equivalent
                # up to complex conjugation and a unitary transformation
                if (nx, ny) > ((-nx) % self.Lx, (-ny) % self.Ly):
                    continue

                parity_reprs = {}
                for monomial in self.basis_reprs:
                    if not self.nonzero_fourier(monomial, (nx, ny)):
                        continue
                    parity = monomial.fermion_parity(spin=True)
                    parity_reprs.setdefault(parity, []).append(monomial)

                for parity in sorted(parity_reprs):
                    self.block_reprs.append(parity_reprs[parity])
                    self.block_momenta.append((nx, ny))

    def _build_psd(self):
        self.psd_blocks = []

        for q, block_basis in zip(self.block_momenta, self.block_reprs):
            psd = PSDConstraints(n_vars=len(self.vars), dim=len(block_basis))

            for row, monomial1 in enumerate(block_basis):
                for col, monomial2 in enumerate(block_basis):
                    expr = {}
                    for dx in range(self.Lx):
                        for dy in range(self.Ly):
                            monomial1r, trans_sign = monomial1.translate(dx, dy)
                            monomial, mul_sign = self._moment(monomial1r, monomial2)

                            if not self._sym_allowed(monomial):
                                continue
                            canon = self._sym_canon(monomial, sign=True)
                            if canon is None:
                                continue
                            key, canon_sign = canon

                            coeff = (
                                self.fourier_phase(q, (dx, dy)) / self.n_sites
                                * trans_sign * mul_sign * canon_sign
                                / monomial.hermitian_phase()
                            )
                            idx = self.var_index[key]
                            expr[idx] = expr.get(idx, 0) + coeff
                            if abs(expr[idx]) < 1e-14:
                                del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))
            self.psd_blocks.append(psd)

    def _add_hamil_wards(self):
        seen_masks = set()
        seen_keys = set()

        for var in self.vars:
            support = var.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 4
                support &= ~(0xf << (4 * site))

                for h_mask, _, _ in self._hamil_terms_at_site[site]:
                    anticomm = (h_mask & var.mask).bit_count() & 1
                    if not anticomm:
                        continue

                    mask = var.mask ^ h_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = MajoranaMonomialSquare(self.Lx, self.Ly, mask)
                    key = cand.trans_canon
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_hamil_ward(cand)
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops['hamil'] += 1

    def _add_number_wards(self, spin: str):
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
                if rem // 2 != spin_offset // 2:
                    continue

                number_mask = 0b11 << (4 * site + spin_offset)
                if (var.mask & number_mask).bit_count() != 1:
                    continue
                mask = var.mask ^ number_mask
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)

                cand = MajoranaMonomialSquare(self.Lx, self.Ly, mask)
                key = cand.trans_canon
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                expr = self._compile_number_ward(cand, spin)
                if expr is None or not expr.terms:
                    continue
                self.affines.add(expr)
                self.ward_ops[f'N{spin}'] += 1

    def _add_spin_wards(self, spin: str):
        assert spin in ('+', '-')
        seen_masks = set()
        seen_keys = set()

        for var in self.vars:
            support = var.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 4
                support &= ~(0xf << (4 * site))

                for rem1, rem2, _ in _SPIN_LADDER_TERMS[spin]:
                    spin_mask = (1 << (4 * site + rem1)) | (1 << (4 * site + rem2))
                    if (var.mask & spin_mask).bit_count() != 1:
                        continue
                    mask = var.mask ^ spin_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = MajoranaMonomialSquare(self.Lx, self.Ly, mask)
                    key = cand.trans_canon
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_spin_ward(cand, spin)
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops[f'S{spin}'] += 1

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))
        for key in self.ward_ops:
            self.ward_ops[key] = 0

        id_key = self._sym_canon(MajoranaMonomialSquare.identity(self.Lx, self.Ly))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

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

        self._add_hamil_wards()
        self._add_number_wards('u')
        self._add_number_wards('d')

        # S^- Ward identities are redundant with S^+ after hermitianization.
        self._add_spin_wards('+')

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _hamil_comm(self, monomial: MajoranaMonomialSquare) -> MajoranaSquareOperator:
        op = MajoranaSquareOperator()
        seen = set()
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4 * site))

            for h_mask, h_coeff, lower_masks in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                anticomm = (h_mask & monomial.mask).bit_count() & 1
                if anticomm:
                    sign = -1 if sum((monomial.mask & lower_mask).bit_count() for lower_mask in lower_masks) & 1 else 1
                    prod = MajoranaMonomialSquare(self.Lx, self.Ly, h_mask ^ monomial.mask)
                    op.add(prod, 2 * h_coeff * sign)
        return op

    def _compile_hamil_ward(self, monomial: MajoranaMonomialSquare) -> LinearExpr | None:
        expr = {}
        seen = set()
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4 * site))

            for h_mask, h_coeff, lower_masks in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                anticomm = (h_mask & monomial.mask).bit_count() & 1
                if not anticomm:
                    continue

                prod = MajoranaMonomialSquare(self.Lx, self.Ly, h_mask ^ monomial.mask)
                if not self._sym_allowed(prod):
                    continue
                canon = self._sym_canon(prod, sign=True)
                if canon is None:
                    continue
                key, canon_sign = canon
                if key not in self.var_index:
                    return None

                sign = -1 if sum((monomial.mask & lower_mask).bit_count() for lower_mask in lower_masks) & 1 else 1
                idx = self.var_index[key]
                expr[idx] = expr.get(idx, 0) + 2 * h_coeff * sign * canon_sign / prod.hermitian_phase()
                if expr[idx] == 0:
                    del expr[idx]

        return LinearExpr(terms=expr, const=0)

    def _compile_number_ward(self, monomial: MajoranaMonomialSquare, spin: str) -> LinearExpr | None:
        assert spin in ('u', 'd')
        spin_offset = 0 if spin == 'u' else 2
        expr = {}

        support = monomial.mask
        while support:
            bit = support & -support
            mode = bit.bit_length() - 1
            site, rem = divmod(mode, 4)
            support ^= bit
            if rem // 2 != spin_offset // 2:
                continue

            lo_bit = 1 << (4 * site + spin_offset)
            number_mask = lo_bit | (lo_bit << 1)
            overlap = monomial.mask & number_mask
            if overlap == 0 or overlap == number_mask:
                continue

            prod = MajoranaMonomialSquare(self.Lx, self.Ly, monomial.mask ^ number_mask)
            if not self._sym_allowed(prod):
                continue
            canon = self._sym_canon(prod, sign=True)
            if canon is None:
                continue
            key, canon_sign = canon
            if key not in self.var_index:
                return None

            sign = -1 if overlap == lo_bit else 1
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + 1j * sign * canon_sign / prod.hermitian_phase()
            if expr[idx] == 0:
                del expr[idx]

        return LinearExpr(terms=expr, const=0)

    def _compile_spin_ward(self, monomial: MajoranaMonomialSquare, spin: str) -> LinearExpr | None:
        assert spin in ('+', '-')
        expr = {}
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4 * site))

            for rem1, rem2, coeff in _SPIN_LADDER_TERMS[spin]:
                mode1 = 4 * site + rem1
                mode2 = 4 * site + rem2
                spin_mask = (1 << mode1) | (1 << mode2)
                if (monomial.mask & spin_mask).bit_count() != 1:
                    continue

                prod = MajoranaMonomialSquare(self.Lx, self.Ly, monomial.mask ^ spin_mask)
                if not self._sym_allowed(prod):
                    continue
                canon = self._sym_canon(prod, sign=True)
                if canon is None:
                    continue
                key, canon_sign = canon
                if key not in self.var_index:
                    return None

                sign = -1 if (
                    (monomial.mask & ((1 << mode1) - 1)).bit_count()
                    + (monomial.mask & ((1 << mode2) - 1)).bit_count()
                ) & 1 else 1
                idx = self.var_index[key]
                expr[idx] = expr.get(idx, 0) + 2 * coeff * sign * canon_sign / prod.hermitian_phase()
                if expr[idx] == 0:
                    del expr[idx]

        return LinearExpr(terms=expr, const=0)

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
            self.energy_ineqs.append(LinearExpr(
                terms=hamil_expr.terms.copy(),
                const=hamil_expr.const - self.e_lb,
            ))
        if self.e_ub is not None:
            self.energy_ineqs.append(LinearExpr(
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
        self._build_affines()

    def descendants(self, monomial: MajoranaMonomialSquare):
        if not hasattr(self, '_descendants_cache'):
            self._descendants_cache = LRU(_CACHE_MAX_SIZE)
        if monomial.mask in self._descendants_cache:
            return self._descendants_cache[monomial.mask]

        entries = {}
        comm_op = self._hamil_comm(monomial)

        for desc, coeff in comm_op.terms.items():
            desc_rep = desc.trans_canon_rep
            shift_x, shift_y, shift_sign = 0, 0, 1
            for dx in range(desc.Lx):
                for dy in range(desc.Ly):
                    shifted, sign = desc_rep.translate(dx, dy)
                    if shifted == desc:
                        shift_x, shift_y, shift_sign = dx, dy, sign
                        break
                else:
                    continue
                break

            key = (desc_rep.mask, shift_x, shift_y)
            entries[key] = entries.get(key, 0) + coeff * shift_sign
            if entries[key] == 0:
                del entries[key]

        descs = [
            (MajoranaMonomialSquare(self.Lx, self.Ly, desc_mask), (shift_x, shift_y), coeff)
            for (desc_mask, shift_x, shift_y), coeff in entries.items()
        ]
        self._descendants_cache[monomial.mask] = descs
        return descs

    def _get_expr_str(self, expr: LinearExpr) -> str:
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
):
    n_sites = Lx * Ly
    if max_degree < 0 or max_degree > 4 * n_sites:
        raise ValueError('max_degree must be between 0 and 4*Lx*Ly')
    if max_support < 0 or max_support > n_sites:
        raise ValueError('max_support must be between 0 and Lx*Ly')

    parsed_windows = []
    for window in windows:
        if len(window) != 2:
            raise ValueError('each window must be a pair [wx, wy]')
        wx, wy = window
        if wx <= 0 or wx > Lx or wy <= 0 or wy > Ly:
            raise ValueError('window sizes must satisfy 1 <= wx <= Lx and 1 <= wy <= Ly')
        parsed_windows.append((wx, wy))
    parsed_windows = sorted(set(parsed_windows))

    identity = MajoranaMonomialSquare.identity(Lx, Ly)
    reps = {identity.trans_canon: identity}
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
                    monomial = MajoranaMonomialSquare(Lx, Ly, raw_mask).trans_canon_rep
                    reps.setdefault(monomial.trans_canon, monomial)

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

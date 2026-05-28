from dataclasses import dataclass
from lru import LRU
import itertools
import numpy as np
import scipy as sp

from operators.majorana import MajoranaMonomial, MajoranaOperator
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData

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


def build_number(params: HubbardParams, spin: str|None = None):
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
    hamil_expr: LinearExpr
    _hamil_terms_at_site: list[list[tuple[int, float | complex, tuple[int, ...]]]]

    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    def __init__(self, params: HubbardParams):
        self.L = params.L
        self.params = params
        self.n_particles = params.n_particles
        self.ward_ops = {'hamil': 0, 'Nu': 0, 'Nd': 0}

        self.hamil_op = build_hamil(params)
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
                self._hamil_terms_at_site[site].append((h_mask, h_coeff, tuple(lower_masks)))

        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op

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

        if monomial.period_sign == -1:
            self._sym_canon_cache[monomial.mask] = None
            return None

        inv_monomial, inv_sign = monomial.invert()
        cands = (
            (monomial.trans_canon, monomial.trans_canon_sign),
            (inv_monomial.trans_canon, inv_sign * inv_monomial.trans_canon_sign),
        )
        canon = min(key for key, _ in cands)
        signs = {sign for key, sign in cands if key == canon}
        if len(signs) > 1:
            self._sym_canon_cache[monomial.mask] = None
            return None

        canon_sign = signs.pop()
        self._sym_canon_cache[monomial.mask] = (canon, canon_sign)
        if sign:
            return canon, canon_sign
        return canon

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
        '''
            return [N_s, O] as a Majorana operator
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
            op.add(prod, 1j * sign / self.L)
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

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))
        for key in self.ward_ops:
            self.ward_ops[key] = 0

        id_key = self._sym_canon(MajoranaMonomial.identity(self.L))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        # fix particle number
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
        # build moments and PSD blocks
        self.basis_reprs = basis_reprs
        self._build_vars()
        self._build_block_reprs()
        self._build_psd()

        # build hamiltonian
        self.hamil_expr = self._compile_expr(self.hamil_op)
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

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
            'hamil_expr': self._get_expr_str(self.hamil_expr),
        }

    def sdp_data(self):
        return SDPData(
            var_cpx = self.var_cpx,
            n_vars = len(self.vars),
            objective = self.hamil_expr,
            psd_blocks = self.psd_blocks,
            affines_mat = self.affines_mat,
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

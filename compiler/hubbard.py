from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from operators.majorana import MajoranaMonomial, MajoranaOperator
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData


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
                L=params.L,
                s=f'{x}{spin}+ {xp1}{spin}-',
                return_sign=True,
            )
            hamil_op.add(monomial, -.5j * params.t * sign / params.L)

            monomial, sign = MajoranaMonomial.from_str(
                L=params.L,
                s=f'{x}{spin}- {xp1}{spin}+',
                return_sign=True,
            )
            hamil_op.add(monomial, .5j * params.t * sign / params.L)

        monomial, sign = MajoranaMonomial.from_str(
            L=params.L,
            s=f'{x}u+ {x}u- {x}d+ {x}d-',
            return_sign=True,
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
                L=params.L,
                s=f'{x}{s}+ {x}{s}-',
                return_sign=True,
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
    # hermitianized Majorana monomials with even fermion parity
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[MajoranaMonomial]

    # moments for generating Ward identities
    ward_index: dict[int, int]
    ward_moments: list[MajoranaMonomial]

    # cache of moment flags
    # var:  SDP variables allowed by symmetries, e.g. 10, 11
    # ward: moment for generating Ward identities, e.g. 01, 11
    # zero: moment pruned by symmetries with zero expectation value, e.g. 00, 01
    _moment_var = 1 << 0
    _moment_ward = 1 << 1
    _moment_flags_cache: dict[int, int]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaOperator
    hamil_expr: LinearExpr

    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    def __init__(self, params: HubbardParams):
        self.L = params.L
        self.params = params
        self.n_particles = params.n_particles
        self.hamil_op = build_hamil(params)
        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op
        self._moment_flags_cache = {}

    @staticmethod
    def _moment(monomial1: MajoranaMonomial, monomial2: MajoranaMonomial) -> tuple[MajoranaMonomial, complex]:
        r'''
            O_1^\dag O_2 with O_1, O_2 Majorana monomials
        '''
        monomial, sign = monomial1.mul(monomial2)
        return monomial, sign * monomial1.dag_phase()

    @staticmethod
    def _translation_pruned(monomial):
        mask = monomial.mask
        for r in range(monomial.L):
            rot_mask, sign = monomial._rotate_l(mask, r, return_sign=True)
            # check if T^\dag(r) O T(r) = -O
            if rot_mask == mask and sign == -1:
                return True
        return False

    def _flag(self, monomial):
        mask = monomial.mask
        if mask in self._moment_flags_cache:
            return self._moment_flags_cache[mask]

        flag = 0
        if monomial.fermion_parity() == -1 or self._translation_pruned(monomial):
            self._moment_flags_cache[mask] = flag
            return flag

        if monomial.k_parity(hermitian=True) == 1:
            flag |= self._moment_var
        else:
            flag |= self._moment_ward
        self._moment_flags_cache[mask] = flag
        return flag

    def _is_var(self, monomial):
        return bool(self._flag(monomial) & self._moment_var)

    def _is_ward(self, monomial):
        return bool(self._flag(monomial) & self._moment_ward)

    def _is_zero(self, monomial):
        return not self._is_var(monomial)

    def _build_moments(self):
        '''
            build SDP variables and moments for generating Ward identities
        '''
        self.var_index = {}
        self.vars = []
        self.ward_index = {}
        self.ward_moments = []

        for monomial1 in self.basis_reprs:
            for r in range(self.L):
                monomial1r = monomial1.translate(r)
                for monomial2 in self.basis_reprs:
                    monomial, _ = self._moment(monomial1r, monomial2)

                    if self._is_var(monomial):
                        key = monomial.canon
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(MajoranaMonomial(L=self.L, mask=key))

                    if self._is_ward(monomial):
                        key = monomial.canon
                        if key not in self.ward_index:
                            self.ward_index[key] = len(self.ward_moments)
                            self.ward_moments.append(MajoranaMonomial(L=self.L, mask=key))

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
            self.block_reprs.append([
                monomial for monomial in self.basis_reprs
                if self.nonzero_fourier(monomial, n)
            ])
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
                        rot_mask, rot_sign = monomial1._rotate_l(monomial1.mask, r, return_sign=True)
                        monomial1r = MajoranaMonomial(L=self.L, mask=rot_mask)
                        monomial, mul_sign = self._moment(monomial1r, monomial2)

                        if self._is_zero(monomial):
                            continue
                        key, canon_sign = monomial.canon, monomial.canon_sign
                        if key not in self.var_index:
                            raise ValueError(f'monomial <{str(monomial)}> is not an SDP variable')

                        # combine Fourier phase, translation/multiplication/canonical signs,
                        # then divide by the hermitian phase so moment variables stay real.
                        coeff = (
                            np.exp(1j * k * r) / self.L
                            * rot_sign * mul_sign * canon_sign
                            / monomial.hermitian_phase()
                        )
                        idx = self.var_index[key]
                        expr[idx] = expr.get(idx, 0) + coeff
                        if abs(expr[idx]) < 1e-12:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))
            self.psd_blocks.append(psd)

    def _hamil_comm(self, monomial: MajoranaMonomial) -> MajoranaOperator:
        if not hasattr(self, '_hamil_terms_by_site'):
            self._hamil_terms_by_site = [[] for _ in range(self.L)]
            for hstr, hcoeff in self.hamil_op.terms.items():
                support = hstr.mask
                h_degree = hstr.mask.bit_count()
                while support:
                    bit = support & -support
                    site = (bit.bit_length() - 1) // 4
                    support &= ~(0xf << (4*site))
                    self._hamil_terms_by_site[site].append((hstr, hcoeff, h_degree))

        op = MajoranaOperator()
        seen = set()
        degree = monomial.mask.bit_count()
        support = monomial.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 4
            support &= ~(0xf << (4*site))

            for hstr, hcoeff, h_degree in self._hamil_terms_by_site[site]:
                if hstr.mask in seen:
                    continue
                seen.add(hstr.mask)

                overlap = (hstr.mask & monomial.mask).bit_count()
                if (h_degree * degree - overlap) & 1:
                    prod, sign = hstr.mul(monomial)
                    op.add(prod, 2 * hcoeff * sign)
        return op

    def _number_comm(self, monomial: MajoranaMonomial, spin: str) -> MajoranaOperator:
        '''
            return [N_s, O] as a Majorana operator
            spin: 'u' or 'd'
        '''
        assert spin in ('u', 'd')
        spin_offset = 0 if spin == 'u' else 2
        op = MajoranaOperator()

        for site in range(self.L):
            pair_mask = 0b11 << (4*site + spin_offset)
            if (monomial.mask & pair_mask).bit_count() != 1:
                continue

            number_term = MajoranaMonomial(self.L, pair_mask)
            prod, sign = number_term.mul(monomial)
            op.add(prod, 1j * sign / self.L)
        return op

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        id_key = MajoranaMonomial.identity(self.L).canon
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
        for monomial in self.ward_moments:
            for comm_op in (
                # time translation (stationarity)
                self._hamil_comm(monomial),
                # U(1)
                self._number_comm(monomial, 'u'),
                self._number_comm(monomial, 'd'),
            ):
                expr = self._compile_expr(comm_op)
                if expr is None:
                    continue
                self.affines.add(expr)

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _compile_expr(self, op: MajoranaOperator) -> LinearExpr | None:
        expr = {}
        for monomial, coeff in op.terms.items():
            if self._is_zero(monomial):
                continue
            key, sign = monomial.canon, monomial.canon_sign
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
        self._build_moments()
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
        entries = {}
        comm_op = self._hamil_comm(monomial)

        for desc, coeff in comm_op.terms.items():
            desc_rep = desc.canon_rep
            s, s_sign = 0, 1
            for shift in range(desc.L):
                mask, sign = desc_rep._rotate_l(desc_rep.mask, shift, return_sign=True)
                if mask == desc.mask:
                    s, s_sign = shift, sign
                    break

            key = (desc_rep.canon, s)
            entries[key] = entries.get(key, 0) + coeff * s_sign
            if entries[key] == 0:
                del entries[key]

        return [
            (MajoranaMonomial(L=monomial.L, mask=desc_canon), s, coeff)
            for (desc_canon, s), coeff in entries.items()
        ]

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
            'ward_moments': len(self.ward_moments),
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
    print(m1.canon)

    print(*m1.mul(m2))
    print(*m2.mul(m1))
    print(*m1.mul(m1))

    op1 = MajoranaOperator({
        MajoranaMonomial.from_str(L=8, s='0u+ 1u+'): 1,
        MajoranaMonomial.from_str(L=8, s='0d+ 0d-'): 2,
    })
    op2 = MajoranaOperator({
        MajoranaMonomial.from_str(L=8, s='1u+ 2d+'): 3,
    })

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

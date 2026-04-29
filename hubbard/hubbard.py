from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData


class MajoranaMonomial:
    '''
        spinful Majorana monomial on a length-L PBC chain (canonical ordered).

        each site carries 4 Majorana modes packed into a 4-bit nibble:
            bit 0: (site, up, +)
            bit 1: (site, up, -)
            bit 2: (site, down, +)
            bit 3: (site, down, -)
    '''
    L: int
    mask: int # 4L-bit
    _canon: int | None
    _canon_sign: int | None

    def __init__(self, L: int, mask: int = 0):
        if L <= 0:
            raise ValueError('L must be positive')
        if mask < 0:
            raise ValueError('mask must be non-negative')
        full = (1 << (4 * L)) - 1
        if mask & ~full:
            raise ValueError('mask exceeds the available 4L Majorana modes')
        self.L = L
        self.mask = mask
        self._canon = None
        self._canon_sign = None

    def __eq__(self, other):
        return self.L == other.L and self.mask == other.mask

    def __hash__(self):
        return hash((self.L, self.mask))

    @classmethod
    def identity(cls, L: int):
        return cls(L=L, mask=0)

    @staticmethod
    def _bit(site: int, spin: int, pm: int) -> int:
        if site < 0:
            raise ValueError('site must be non-negative')
        if spin not in (0, 1):
            raise ValueError('spin must be 0 (up) or 1 (down)')
        if pm not in (0, 1):
            raise ValueError('pm must be 0 (+) or 1 (-)')
        return 1 << (4*site + 2*spin + pm)

    @staticmethod
    def _parse_token(token: str):
        if len(token) < 3:
            raise ValueError(f'invalid Majorana token: {token}')

        spin_map = {'u': 0, 'd': 1}
        pm_map = {'+': 0, '-': 1}
        site = token[:-2]
        spin = token[-2]
        pm = token[-1]
        if not site.isdigit():
            raise ValueError(f'invalid site in Majorana token: {token}')
        if spin not in spin_map:
            raise ValueError(f'invalid spin in Majorana token: {token}')
        if pm not in pm_map:
            raise ValueError(f'invalid +/- label in Majorana token: {token}')

        return int(site), spin_map[spin], pm_map[pm]

    @classmethod
    def from_str(cls, L: int, s: str, return_sign: bool = False):
        '''
            build an ordered product by right-multiplying degree-1 monomials from left to right
        '''
        s = s.strip()
        if not s or s == 'I':
            product = cls.identity(L=L)
            return (product, 1) if return_sign else product

        tokens = s.split()
        mask = 0
        total_sign = 1

        for token in tokens:
            bit = cls._bit(*cls._parse_token(token))
            if (mask >> bit.bit_length()).bit_count() % 2 == 1:
                total_sign = -total_sign
            mask ^= bit

        return (cls(L=L, mask=mask), total_sign) if return_sign else cls(L=L, mask=mask)

    def _rotate_l(self, mask: int, shift: int, return_sign: bool = False) -> int | tuple[int, int]:
        shift %= self.L
        if shift == 0:
            return (mask, 1) if return_sign else mask

        full = (1 << (4 * self.L)) - 1
        rot = 4 * shift
        rot_mask = ((mask << rot) | (mask >> (4 * self.L - rot))) & full
        if not return_sign:
            return rot_mask

        cutoff = 4 * (self.L - shift)
        n_wrap = (mask >> cutoff).bit_count()
        n_keep = mask.bit_count() - n_wrap
        sign = -1 if (n_wrap * n_keep) % 2 else 1
        return rot_mask, sign

    def translate(self, shift: int):
        monomial = MajoranaMonomial.__new__(MajoranaMonomial)
        monomial.L = self.L
        monomial.mask = self._rotate_l(self.mask, shift)
        monomial._canon = self._canon
        monomial._canon_sign = self._canon_sign
        return monomial

    def canon(self, return_sign: bool = False) -> int | tuple[int, int]:
        if self._canon is not None:
            return (self._canon, self._canon_sign) if return_sign else self._canon

        canon = self.mask
        sign = 1
        for shift in range(1, self.L):
            cand, cand_sign = self._rotate_l(self.mask, shift, return_sign=True)
            if cand < canon:
                canon = cand
                sign = cand_sign

        self._canon = canon
        self._canon_sign = sign
        return (canon, sign) if return_sign else canon

    def mul(self, other) -> tuple['MajoranaMonomial', int]:
        if self.L != other.L:
            raise ValueError('cannot multiply Majorana monomials with different L')

        sign = 1
        right = other.mask
        while right:
            bit = right & -right
            if (self.mask >> bit.bit_length()).bit_count() % 2 == 1:
                sign = -sign
            right ^= bit

        return MajoranaMonomial(L=self.L, mask=self.mask^other.mask), sign

    def degree(self) -> int:
        '''
            degree of the monomial
        '''
        return self.mask.bit_count()

    def dag_phase(self):
        '''
            phase acquired after hermitian operation
        '''
        degree = self.degree()
        return 1 if ((degree*(degree-1)) // 2) % 2 == 0 else -1

    def hermitian_phase(self):
        '''
            phase needed to make Majorana monomial a hermitian
        '''
        return 1 if self.dag_phase() == 1 else 1j

    def __str__(self):
        if self.mask == 0:
            return 'I'
        spin_names = ('u', 'd')
        pm_names = ('+', '-')
        parts = []
        for mode in range(4 * self.L):
            if self.mask & (1 << mode):
                site, rem = divmod(mode, 4)
                spin, pm = divmod(rem, 2)
                parts.append(f'{site}{spin_names[spin]}{pm_names[pm]}')
        return ' '.join(parts)


class MajoranaOperator:
    L: int | None
    terms: dict[MajoranaMonomial, float|complex]

    def __init__(self, terms=None):
        self.L = None # L is None only for zero operator
        self.terms = {}
        if terms is None:
            return
        for monomial, coeff in terms.items():
            self.add(monomial, coeff)

    def __str__(self):
        if not self.terms:
            return '0'
        parts = []
        for monomial, coeff in self.terms.items():
            parts.append(f'{coeff}*({monomial})')
        return ' + '.join(parts)

    def copy(self):
        op = MajoranaOperator()
        op.L = self.L
        op.terms = self.terms.copy()
        return op

    def add(self, monomial: MajoranaMonomial, coeff: float|complex):
        if coeff == 0:
            return
        if self.L is None:
            self.L = monomial.L
        elif monomial.L != self.L:
            raise ValueError('Majorana monomials in an operator must have the same L')

        self.terms[monomial] = self.terms.get(monomial, 0) + coeff
        if self.terms[monomial] == 0:
            del self.terms[monomial]
        if not self.terms:
            self.L = None

    def __add__(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = self.copy()
        for monomial, coeff in other.terms.items():
            op.add(monomial, coeff)
        return op

    def __sub__(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = self.copy()
        for monomial, coeff in other.terms.items():
            op.add(monomial, -coeff)
        return op

    def __neg__(self):
        op = MajoranaOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, -coeff)
        return op

    def __rmul__(self, scalar):
        op = MajoranaOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, scalar * coeff)
        return op

    def mul(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = MajoranaOperator()
        for monomial1, coeff1 in self.terms.items():
            for monomial2, coeff2 in other.terms.items():
                monomial, sign = monomial1.mul(monomial2)
                coeff = coeff1 * coeff2 * sign
                op.add(monomial, coeff)
        return op

    def dag(self):
        op = MajoranaOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, coeff.conjugate() * monomial.dag_phase())
        return op

    def commutator(self, other):
        return self.mul(other) - other.mul(self)


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
            hamil_op.add(monomial, -.5j * params.t * sign)

            monomial, sign = MajoranaMonomial.from_str(
                L=params.L,
                s=f'{x}{spin}- {xp1}{spin}+',
                return_sign=True,
            )
            hamil_op.add(monomial, .5j * params.t * sign)

        monomial, sign = MajoranaMonomial.from_str(
            L=params.L,
            s=f'{x}u+ {x}u- {x}d+ {x}d-',
            return_sign=True,
        )
        hamil_op.add(monomial, -.25 * params.U * sign)

    return hamil_op


def build_number(params: HubbardParams, spin: str|None = None):
    assert spin in (None, 'u', 'd')
    spins = ('u', 'd') if spin is None else (spin,)
    number_op = MajoranaOperator()
    for s in spins:
        number_op.add(MajoranaMonomial.identity(params.L), .5 * params.L)
        for x in range(params.L):
            monomial, sign = MajoranaMonomial.from_str(
                L=params.L,
                s=f'{x}{s}+ {x}{s}-',
                return_sign=True,
            )
            number_op.add(monomial, .5j * sign)
    return number_op


class HubbardCompiler:
    params: HubbardParams

    basis_reprs: list[MajoranaMonomial]

    # moment variables are real expectations of
    # hermitianized Majorana monomials with even fermion parity
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[MajoranaMonomial]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: MajoranaOperator
    hamil_expr: LinearExpr

    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    _pruned_cache: set[int]
    _unpruned_cache: set[int]

    def __init__(self, params: HubbardParams):
        self.L = params.L
        self.t = params.t
        self.U = params.U
        self.n_particles = params.n_particles
        self.hamil_op = build_hamil(params)
        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op

        self._pruned_cache = set()
        self._unpruned_cache = set()

    @staticmethod
    def _moment(monomial1: MajoranaMonomial, monomial2: MajoranaMonomial) -> tuple[MajoranaMonomial, complex]:
        r'''
            O_1^\dag O_2 with O_1, O_2 Majorana monomials
        '''
        monomial, sign = monomial1.mul(monomial2)
        return monomial, sign * monomial1.dag_phase()

    def _pruned(self, monomial) -> bool:
        '''
            return whether the monomial variable is pruned by symmetry
        '''
        mask = monomial.mask
        if mask in self._pruned_cache:
            return True
        if mask in self._unpruned_cache:
            return False

        # Fermion parity
        if monomial.degree() % 2:
            self._pruned_cache.add(mask)
            return True

        # translation invariance
        for r in range(monomial.L):
            rot_mask, sign = monomial._rotate_l(mask, r, return_sign=True)
            if rot_mask == mask and sign == -1:
                # check if T^\dag(r) O T(r) = -O
                self._pruned_cache.add(mask)
                return True

        self._unpruned_cache.add(mask)
        return False

    def _build_vars(self):
        self.var_index = {}
        self.vars = []

        for monomial1 in self.basis_reprs:
            for r in range(self.L):
                monomial1r = monomial1.translate(r)
                for monomial2 in self.basis_reprs:
                    monomial, _ = self._moment(monomial1r, monomial2)
                    if self._pruned(monomial):
                        continue

                    key = monomial.canon()
                    if key not in self.var_index:
                        self.var_index[key] = len(self.vars)
                        self.vars.append(MajoranaMonomial(L=self.L, mask=key))

    def _build_psd(self):
        '''
            build momentum PSD blocks M(k)
        '''
        self.psd_blocks = []
        dim = len(self.basis_reprs)
        for n in range(self.L):
            k = 2*np.pi * n / self.L
            psd = PSDConstraints(n_vars=len(self.vars), dim=dim)
            for row, monomial1 in enumerate(self.basis_reprs):
                for col, monomial2 in enumerate(self.basis_reprs):
                    expr = {}
                    for r in range(self.L):
                        rot_mask, rot_sign = monomial1._rotate_l(monomial1.mask, r, return_sign=True)
                        monomial1r = MajoranaMonomial(L=self.L, mask=rot_mask)
                        monomial, mul_sign = self._moment(monomial1r, monomial2)
                        if self._pruned(monomial):
                            continue
                        key, canon_sign = monomial.canon(return_sign=True)

                        # combine Fourier phase, translation/multiplication/canonical signs,
                        # then divide by the hermitian phase so moment variables stay real.
                        coeff = (
                            np.exp(1j * k * r)
                            * rot_sign * mul_sign * canon_sign
                            / monomial.hermitian_phase()
                        )
                        idx = self.var_index[key]
                        expr[idx] = expr.get(idx, 0) + coeff
                        if expr[idx] == 0:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))
            self.psd_blocks.append(psd)

    def _compile_expr(self, op: MajoranaOperator) -> LinearExpr | None:
        expr = {}
        for monomial, coeff in op.terms.items():
            if self._pruned(monomial):
                continue
            key, sign = monomial.canon(return_sign=True)
            if key not in self.var_index:
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff * sign / monomial.hermitian_phase()
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def compile(self, basis_reprs: list[MajoranaMonomial]):
        self._pruned_cache = set()
        self._unpruned_cache = set()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ build variables and psd blocks
        self.basis_reprs = basis_reprs
        self._build_vars()
        self._build_psd()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ build hamiltonian
        self.hamil_expr = self._compile_expr(self.hamil_op)
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ build affine constraints
        self.affines = AffineConstraints(n_vars=len(self.vars))
        id_key = MajoranaMonomial.identity(self.L).canon()
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        # fix particle number
        if self.n_particles is not None:
            number_shift = self.number_op.copy()
            number_shift.add(MajoranaMonomial.identity(self.L), -self.n_particles)

            number_expr = self._compile_expr(number_shift)
            if number_expr is None:
                raise ValueError('current basis cannot represent the particle number operator')
            self.affines.add(number_expr)

            number_var_expr = self._compile_expr(number_shift.mul(number_shift))
            if number_var_expr is None:
                raise ValueError('current basis cannot represent the particle number variance operator')
            self.affines.add(number_var_expr)

        # Ward identities
        generators = (
            # time translation (stationarity)
            self.hamil_op,
            # U(1)
            self.number_up_op,
            self.number_dn_op,
        )
        for generator in generators:
            # parity-even symmetry generators preserve monomial parity, so only parity-even
            # moment variables can produce nontrivial constraints after odd moments are removed.
            for monomial in self.vars:
                comm_op = generator.commutator(MajoranaOperator({monomial: 1}))
                expr = self._compile_expr(comm_op)
                if expr is None:
                    continue
                self.affines.add(expr)

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-10)

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
            'L': self.L,
            'basis_reprs': len(self.basis_reprs),
            'vars': len(self.vars),
            'psd_blocks': len(self.psd_blocks),
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
def load_basis_reprs(L: int, max_degree: int, max_support: int, max_diameter: int):
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
    print(m1.canon())

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
    compiler.compile(basis_reprs=load_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    ))
    print(compiler.summary())

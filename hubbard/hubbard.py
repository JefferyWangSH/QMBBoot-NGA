from dataclasses import dataclass
import numpy as np

from sdp import LinearExpr, SDPData


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

    def degree(self) -> int:
        '''
            degree of the monomial
        '''
        return self.mask.bit_count()

    def local_mask(self, site: int) -> int:
        '''
            4-bit local mask 
        '''
        if not 0 <= site < self.L:
            raise ValueError('site out of range')
        return (self.mask >> (4 * site)) & 0xF

    def support_sites(self) -> tuple[int, ...]:
        return tuple(site for site in range(self.L) if self.local_mask(site) != 0)

    def support(self) -> int:
        return len(self.support_sites())

    def diameter(self) -> int:
        sites = self.support_sites()
        if len(sites) <= 1:
            return 0

        diam = 0
        for i, site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                dist = abs(site1 - site2)
                diam = max(diam, min(dist, self.L-dist))
        return diam

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

    def _iter_bits(self):
        for mode in range(4 * self.L):
            bit = 1 << mode
            if self.mask & bit:
                yield bit

    def __str__(self):
        if self.mask == 0:
            return 'I'

        spin_names = ('u', 'd')
        pm_names = ('+', '-')
        parts = []
        for bit in self._iter_bits():
            mode = bit.bit_length() - 1
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
    basis_full: list[MajoranaMonomial]

    # moment variables are real expectations of
    # hermitianized Majorana monomials with even fermion parity
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[MajoranaMonomial]
    M: list[list[LinearExpr]]

    constraints: list[LinearExpr]
    constraints_rank: int

    hamil_op: MajoranaOperator
    hamil_expr: LinearExpr

    number_op: MajoranaOperator
    number_up_op: MajoranaOperator
    number_dn_op: MajoranaOperator

    def __init__(self, params: HubbardParams):
        self.L = params.L
        self.t = params.t
        self.U = params.U
        self.n_particles = params.n_particles
        self.hamil_op = build_hamil(params)
        self.number_up_op = build_number(params, spin='u')
        self.number_dn_op = build_number(params, spin='d')
        self.number_op = self.number_up_op + self.number_dn_op

    @staticmethod
    def _moment(mono1: MajoranaMonomial, mono2: MajoranaMonomial) -> tuple[MajoranaMonomial, complex]:
        r'''
            O_1^\dag O_2 with O_1, O_2 Majorana monomials
        '''
        monomial, sign = mono1.mul(mono2)
        return monomial, sign * mono1.dag_phase()

    def _build_moments(self):
        self.var_index = {}
        self.vars = []
        self.M = []

        for monomial1 in self.basis_full:
            row = []
            for monomial2 in self.basis_full:
                monomial, phase = self._moment(monomial1, monomial2)
                if monomial.degree() % 2:
                    # fermion parity odd
                    row.append(LinearExpr(terms={}, const=0))
                    continue

                key, sign = monomial.canon(return_sign=True)
                if key not in self.var_index:
                    self.var_index[key] = len(self.vars)
                    self.vars.append(MajoranaMonomial(L=self.L, mask=key))
                # sign compensates the swap sign generated during canonicalization;
                # to make the optimization variables real,
                # additional hermitian phase is factored out to make Majorana monomial a hermitian.
                row.append(LinearExpr(
                    terms={self.var_index[key]: phase * sign / monomial.hermitian_phase()},
                    const=0,
                ))
            self.M.append(row)

    def _compile_expr(self, op: MajoranaOperator) -> LinearExpr | None:
        expr = {}
        for monomial, coeff in op.terms.items():
            key, sign = monomial.canon(return_sign=True)
            if key not in self.var_index:
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff * sign / monomial.hermitian_phase()
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def compile(self, basis_reprs: list[MajoranaMonomial]):
        self.basis_reprs = basis_reprs
        self.basis_full = []
        basis_seen = set()
        for monomial in self.basis_reprs:
            for shift in range(self.L):
                monomial_shift = monomial.translate(shift)
                if monomial_shift not in basis_seen:
                    basis_seen.add(monomial_shift)
                    self.basis_full.append(monomial_shift)

        self._build_moments()

        self.hamil_expr = self._compile_expr(self.hamil_op)
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        self.constraints = []
        id_key = MajoranaMonomial.identity(self.L).canon()
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.constraints.append(LinearExpr(
            terms={self.var_index[id_key]: 1.},
            const=-1.,
        ))

        if self.n_particles is not None:
            number_shift = self.number_op.copy()
            number_shift.add(MajoranaMonomial.identity(self.L), -self.n_particles)

            number_expr = self._compile_expr(number_shift)
            if number_expr is None:
                raise ValueError('current basis cannot represent the particle number operator')
            self.constraints.append(number_expr)

            number_var_expr = self._compile_expr(number_shift.mul(number_shift))
            if number_var_expr is None:
                raise ValueError('current basis cannot represent the particle number variance operator')
            self.constraints.append(number_var_expr)

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
                if expr.terms or expr.const != 0:
                    self.constraints.append(expr)

        if not self.constraints:
            self.constraints_rank = 0
        else:
            mat = np.zeros((len(self.constraints), len(self.vars)), dtype=np.complex128)
            for row, expr in enumerate(self.constraints):
                for idx, coeff in expr.terms.items():
                    mat[row, idx] = coeff
            self.constraints_rank = int(np.linalg.matrix_rank(mat))

        # prune linear-dependent constraints
        ...

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
            'basis_full': len(self.basis_full),
            'vars': len(self.vars),
            'constraints': len(self.constraints),
            'constraints_rank': self.constraints_rank,
            'hamil_expr': self._get_expr_str(self.hamil_expr),
        }

    def sdp_data(self):
        return SDPData(
            var_cpx=self.var_cpx,
            n_vars=len(self.vars),
            M=self.M,
            constraints=self.constraints,
            objective=self.hamil_expr,
        )


'''
    basis_reprs helper
'''
def load_basis_reprs(L:int, type='local'):
    assert type in ('local',)
    if type == 'local':
        return [
            MajoranaMonomial(L=L, mask=mask)
            for mask in range(1 << 4)
        ]
    ...


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    m1 = MajoranaMonomial.from_str(L=8, s='0u+ 0u- 2d+')
    m2 = MajoranaMonomial.from_str(L=8, s='0u+ 3d-')

    print(m1)
    print(m1.mask)
    print(m1.degree())
    print(m1.support_sites())
    print(m1.support())
    print(m1.diameter())
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

    compiler.compile(basis_reprs=load_basis_reprs(params.L, 'local'))
    print(compiler.summary())

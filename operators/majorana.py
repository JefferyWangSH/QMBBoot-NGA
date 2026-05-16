from functools import cached_property

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
    canon: int
    canon_sign: int
    canon_rep: 'MajoranaMonomial'
    period: int
    period_sign: int
    _canon_data: tuple[int, int]
    _period_data: tuple[int, int]

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
        return monomial

    @cached_property
    def _canon_data(self) -> tuple[int, int]:
        canon = self.mask
        sign = 1
        for shift in range(1, self.L):
            cand, cand_sign = self._rotate_l(self.mask, shift, return_sign=True)
            if cand < canon:
                canon = cand
                sign = cand_sign
        return canon, sign

    @cached_property
    def canon(self) -> int:
        return self._canon_data[0]

    @cached_property
    def canon_sign(self) -> int:
        return self._canon_data[1]

    @cached_property
    def canon_rep(self):
        return MajoranaMonomial(L=self.L, mask=self.canon)

    @cached_property
    def _period_data(self) -> tuple[int, int]:
        for shift in range(1, self.L):
            mask, sign = self._rotate_l(self.mask, shift, return_sign=True)
            if mask == self.mask:
                return shift, sign
        return self.L, 1

    @cached_property
    def period(self) -> int:
        return self._period_data[0]

    @cached_property
    def period_sign(self) -> int:
        return self._period_data[1]

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

    def fermion_parity(self):
        return 1 - 2 * (self.degree() % 2)

    def k_parity(self, hermitian=True):
        mask = sum(1 << mode for mode in range(1, 4*self.L, 2))
        # number of \gamma_2
        cnt = (self.mask & mask).bit_count()
        # patch from hermitianization
        if hermitian and self.dag_phase() == -1:
            cnt += 1
        return 1 - 2 * (cnt % 2)

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
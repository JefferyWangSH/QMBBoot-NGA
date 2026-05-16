from functools import cached_property

class PauliString:
    L: int
    mask: int # 2L-bit, each site uses I=00, X=01, Z=10, Y=11

    # canonical representation, unique as the translation-invariant representative
    canon: int
    canon_rep: 'PauliString'
    period: int

    def __init__(self, L: int, mask: int = 0):
        if L <= 0:
            raise ValueError('L must be positive')
        if mask < 0:
            raise ValueError('mask must be non-negative')
        full = (1 << (2*L)) - 1
        if mask & ~full:
            raise ValueError('mask exceeds the available 2L Pauli bits')
        self.L = L
        self.mask = mask

    @classmethod
    def from_str(cls, pstr: str):
        mask = 0
        for i, pauli in enumerate(pstr):
            if pauli == 'I':
                code = 0
            elif pauli == 'X':
                code = 1
            elif pauli == 'Z':
                code = 2
            elif pauli == 'Y':
                code = 3
            else:
                raise ValueError(f'invalid Pauli operator: {pauli}')
            mask |= code << (2*i)
        return cls(L=len(pstr), mask=mask)

    def _rotate_l(self, mask: int, shift: int) -> int:
        shift %= self.L
        if shift == 0:
            return mask
        full = (1 << (2*self.L)) - 1
        rot = 2*shift
        return ((mask << rot) | (mask >> (2*self.L - rot))) & full

    @cached_property
    def canon(self) -> int:
        canon = self.mask
        for shift in range(1, self.L):
            cand = self._rotate_l(self.mask, shift)
            if cand < canon:
                canon = cand
        return canon

    @cached_property
    def canon_rep(self):
        return PauliString(self.L, self.canon)

    @cached_property
    def period(self) -> int:
        for shift in range(1, self.L):
            if self._rotate_l(self.mask, shift) == self.mask:
                return shift
        return self.L

    def translate(self, shift: int):
        return PauliString(self.L, self._rotate_l(self.mask, shift))

    def __eq__(self, other):
        return self.L == other.L and self.mask == other.mask

    def __hash__(self):
        return hash((self.L, self.mask))

    def __str__(self):
        chars = []
        paulis = 'IXZY'
        for i in range(self.L):
            chars.append(paulis[(self.mask >> (2*i)) & 3])
        return ''.join(chars)

    def dag(self):
        return self

    def mul(self, other) -> tuple["PauliString", float|complex]:
        assert self.L == other.L
        mask = self.mask ^ other.mask

        phase = 1.
        support = self.mask | other.mask

        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            a = (self.mask >> (2*site)) & 3
            b = (other.mask >> (2*site)) & 3

            if a == 1 and b == 2:   # XZ = -iY
                phase *= -1j
            elif a == 2 and b == 1: # ZX = iY
                phase *= 1j
            elif a == 1 and b == 3: # XY = iZ
                phase *= 1j
            elif a == 3 and b == 1: # YX = -iZ
                phase *= -1j
            elif a == 2 and b == 3: # ZY = -iX
                phase *= -1j
            elif a == 3 and b == 2: # YZ = iX
                phase *= 1j

            support &= ~(3 << (2*site))

        return PauliString(self.L, mask), phase

    def parity(self):
        # +1 for K even and -1 for K odd
        y_count = sum(
            1 for i in range(self.L)
            if ((self.mask >> (2*i)) & 3) == 3
        )
        return 1 - 2 * int(y_count % 2)


class PauliOperator:
    L: int | None
    terms: dict[PauliString, float|complex]

    def __init__(self, terms=None):
        self.L = None # L is None only for zero operator
        self.terms = {}
        if terms is None:
            return
        for pstr, coeff in terms.items():
            self.add(pstr, coeff)

    def __str__(self):
        if not self.terms:
            return '0'
        return ' + '.join([
            f'{coeff}*{str(pstr)}' for pstr, coeff in self.terms.items()
        ])

    def copy(self):
        op = PauliOperator()
        op.L = self.L
        op.terms = self.terms.copy()
        return op

    def add(self, pstr: PauliString, coeff: float|complex):
        if coeff == 0:
            return
        if self.L is None:
            self.L = pstr.L
        elif pstr.L != self.L:
            raise ValueError('Pauli strings in an operator must have the same L')
        self.terms[pstr] = self.terms.get(pstr, 0) + coeff
        if self.terms[pstr] == 0:
            del self.terms[pstr]
        if not self.terms:
            self.L = None

    def __add__(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = self.copy()
        for pstr, coeff in other.terms.items():
            op.add(pstr, coeff)
        return op

    def __sub__(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = self.copy()
        for pstr, coeff in other.terms.items():
            op.add(pstr, -coeff)
        return op

    def __neg__(self):
        op = PauliOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr, -coeff)
        return op

    def __rmul__(self, scalar):
        op = PauliOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr, scalar * coeff)
        return op

    def mul(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = PauliOperator()
        for pstr1, coeff1 in self.terms.items():
            for pstr2, coeff2 in other.terms.items():
                pstr, phase = pstr1.mul(pstr2)
                coeff = coeff1 * coeff2 * phase
                op.add(pstr, coeff)
        return op

    def dag(self):
        op = PauliOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr.dag(), coeff.conjugate())
        return op

    def commutator(self, other):
        return self.mul(other) - other.mul(self)
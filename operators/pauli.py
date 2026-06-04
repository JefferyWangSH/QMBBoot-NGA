from functools import cached_property

_PAULI_CODE = {'X': 1, 'Y': 3, 'Z': 2}
_PAULI_PHASE = (1., 1j, -1., -1j)
_PAULI_MUL_PHASE_POWER = (
    (0, 0, 0, 0),
    (0, 0, 3, 1),
    (0, 1, 0, 3),
    (0, 3, 1, 0),
)


class PauliString:
    L: int
    mask: int # 2L-bit, each site uses I=00, X=01, Z=10, Y=11
    trans_canon: int
    trans_canon_rep: 'PauliString'
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
                continue
            try:
                code = _PAULI_CODE[pauli]
            except KeyError:
                raise ValueError(f'invalid Pauli operator: {pauli}') from None
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
    def trans_canon(self) -> int:
        canon = self.mask
        for shift in range(1, self.L):
            cand = self._rotate_l(self.mask, shift)
            if cand < canon:
                canon = cand
        return canon

    @cached_property
    def trans_canon_rep(self):
        return PauliString(self.L, self.trans_canon)

    @cached_property
    def period(self) -> int:
        for shift in range(1, self.L):
            if self._rotate_l(self.mask, shift) == self.mask:
                return shift
        return self.L

    def translate(self, shift: int):
        return PauliString(self.L, self._rotate_l(self.mask, shift))

    def invert(self):
        # lattice inversion i <-> -i mod L
        mask = 0
        support = self.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            code = (self.mask >> (2*site)) & 3
            mask |= code << (2*((-site) % self.L))
            support &= ~(3 << (2*site))
        return PauliString(self.L, mask)

    def permute(self, perm: tuple[str, str, str]):
        code_map = {1: _PAULI_CODE[perm[0]], 3: _PAULI_CODE[perm[1]], 2: _PAULI_CODE[perm[2]]}
        mask = 0
        support = self.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            old_code = (self.mask >> (2*site)) & 3
            mask |= code_map[old_code] << (2*site)
            support &= ~(3 << (2*site))
        return PauliString(self.L, mask)

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
        support = self.mask | other.mask
        phase_power = 0

        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            a = (self.mask >> (2*site)) & 3
            b = (other.mask >> (2*site)) & 3
            phase_power += _PAULI_MUL_PHASE_POWER[a][b]
            support &= ~(3 << (2*site))

        return PauliString(self.L, mask), _PAULI_PHASE[phase_power & 3]

    def parity(self):
        # 0 for K even and 1 for K odd
        y_count = sum(
            1 for i in range(self.L)
            if ((self.mask >> (2*i)) & 3) == 3
        )
        return y_count % 2

    def sign_charge(self):
        '''
            sign-symmetry charge
        '''
        charge = [0, 0, 0]
        support = self.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            code = (self.mask >> (2*site)) & 3
            if code == 1:
                charge[0] ^= 1
            elif code == 3:
                charge[1] ^= 1
            elif code == 2:
                charge[2] ^= 1
            support &= ~(3 << (2*site))
        return tuple(charge)

    def pi_rot_charge(self):
        '''
            pi spin-rotation charge, map to sign-symmetry charge as

                ++: 000/111, +-: 001/110,
                -+: 100/011, --: 010/101
        '''
        nx, ny, nz = self.sign_charge()
        return ((nx + ny) & 1, (ny + nz) & 1)


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

    def __eq__(self, other):
        return self.L == other.L and self.terms == other.terms

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
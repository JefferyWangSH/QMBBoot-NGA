from .majorana_square_jit import MajoranaMonomialSquare

class MajoranaSquareOperator:
    shape: tuple[int, int] | None
    terms: dict[MajoranaMonomialSquare, float | complex]

    def __init__(self, terms=None):
        self.shape = None  # shape is None only for the zero operator
        self.terms = {}
        if terms is None:
            return
        for monomial, coeff in terms.items():
            self.add(monomial, coeff)

    @property
    def Lx(self) -> int | None:
        return None if self.shape is None else self.shape[0]

    @property
    def Ly(self) -> int | None:
        return None if self.shape is None else self.shape[1]

    def __str__(self):
        if not self.terms:
            return '0'
        parts = []
        for monomial, coeff in self.terms.items():
            parts.append(f'{coeff}*({monomial})')
        return ' + '.join(parts)

    def __eq__(self, other):
        return self.shape == other.shape and self.terms == other.terms

    def copy(self):
        op = MajoranaSquareOperator()
        op.shape = self.shape
        op.terms = self.terms.copy()
        return op

    def add(self, monomial: MajoranaMonomialSquare, coeff: float | complex):
        if coeff == 0:
            return
        shape = (monomial.Lx, monomial.Ly)
        if self.shape is None:
            self.shape = shape
        elif shape != self.shape:
            raise ValueError('Majorana monomials in an operator must have the same Lx, Ly')

        self.terms[monomial] = self.terms.get(monomial, 0) + coeff
        if self.terms[monomial] == 0:
            del self.terms[monomial]
        if not self.terms:
            self.shape = None

    def __add__(self, other):
        assert self.shape is None or other.shape is None or self.shape == other.shape
        op = self.copy()
        for monomial, coeff in other.terms.items():
            op.add(monomial, coeff)
        return op

    def __sub__(self, other):
        assert self.shape is None or other.shape is None or self.shape == other.shape
        op = self.copy()
        for monomial, coeff in other.terms.items():
            op.add(monomial, -coeff)
        return op

    def __neg__(self):
        op = MajoranaSquareOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, -coeff)
        return op

    def __rmul__(self, scalar):
        op = MajoranaSquareOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, scalar * coeff)
        return op

    def mul(self, other):
        assert self.shape is None or other.shape is None or self.shape == other.shape
        op = MajoranaSquareOperator()
        for monomial1, coeff1 in self.terms.items():
            for monomial2, coeff2 in other.terms.items():
                monomial, sign = monomial1.mul(monomial2)
                op.add(monomial, coeff1 * coeff2 * sign)
        return op

    def dag(self):
        op = MajoranaSquareOperator()
        for monomial, coeff in self.terms.items():
            op.add(monomial, coeff.conjugate() * monomial.dag_phase())
        return op

    def commutator(self, other):
        return self.mul(other) - other.mul(self)

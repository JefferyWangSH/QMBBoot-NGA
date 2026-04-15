from dataclasses import dataclass
import numpy as np

from sdp import LinearExpr

'''
    length-L transverse Ising chain (PBC)
'''

class Operator:
    L: int      # bit length
    x_mask: int # bit representation of x operators
    z_mask: int # bit representation of z operators

    # ocanonical representation, unique as the translation-invariant representative (TIR)
    _canon: int | None

    def __init__(self, op_str: str):
        self.L = len(op_str)
        self.x_mask = 0
        self.z_mask = 0
        for i, op in enumerate(op_str):
            if op in 'XY':
                self.x_mask |= 1 << i
            if op in 'ZY':
                self.z_mask |= 1 << i
        self._canon = None

    @classmethod
    def _from_masks(cls, L: int, x_mask: int, z_mask: int):
        op = cls.__new__(cls)
        op.L = L
        op.x_mask = x_mask
        op.z_mask = z_mask
        op._canon = None
        return op
    
    def _rotate_l(self, mask: int, shift: int) -> int:
        shift %= self.L
        full = (1 << self.L) - 1
        return ((mask << shift) | (mask >> (self.L - shift))) & full

    def _pack(self, x_mask: int, z_mask: int) -> int:
        return (x_mask << self.L) | z_mask

    def canon(self) -> int:
        if self._canon is not None:
            return self._canon

        self._canon = self._pack(self.x_mask, self.z_mask)
        for shift in range(1, self.L):
            x_rot = self._rotate_l(self.x_mask, shift)
            z_rot = self._rotate_l(self.z_mask, shift)
            cand = self._pack(x_rot, z_rot)
            if cand < self._canon:
                self._canon = cand
        return self._canon

    def translate(self, shift: int):
        op = Operator.__new__(Operator)
        op.L = self.L
        op.x_mask = self._rotate_l(self.x_mask, shift)
        op.z_mask = self._rotate_l(self.z_mask, shift)
        op._canon = self._canon
        return op

    def __eq__(self, other):
        return (
            self.L == other.L and
            self.x_mask == other.x_mask and
            self.z_mask == other.z_mask
        )

    def __hash__(self):
        return hash((self.L, self.x_mask, self.z_mask))

    def __str__(self):
        chars = []
        for i in range(self.L):
            x = (self.x_mask >> i) & 1
            z = (self.z_mask >> i) & 1
            if x == 0 and z == 0:
                chars.append('I')
            elif x == 1 and z == 0:
                chars.append('X')
            elif x == 0 and z == 1:
                chars.append('Z')
            else:
                chars.append('Y')
        return ''.join(chars)
    
    def dag(self):
        return self
    
    def mul(self, other) -> tuple["Operator", complex]:
        assert self.L == other.L
        x_mask = self.x_mask ^ other.x_mask
        z_mask = self.z_mask ^ other.z_mask

        phase = 1.0 + 0.0j
        support = self.x_mask | self.z_mask | other.x_mask | other.z_mask

        while support:
            bit = support & -support
            a = ((self.x_mask & bit) != 0) + 2 * ((self.z_mask & bit) != 0)
            b = ((other.x_mask & bit) != 0) + 2 * ((other.z_mask & bit) != 0)

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

            support ^= bit

        return Operator._from_masks(self.L, x_mask, z_mask), phase


class OperatorSum:
    L: int
    terms: dict[Operator, complex]

    def __init__(self, terms=None):
        self.terms = {}
        if terms is not None:
            for op, coeff in terms.items():
                if coeff != 0:
                    self.terms[op] = coeff

    def __str__(self):
        if not self.terms:
            return '0'
        return ' + '.join([
            f'{coeff}*{op}' for op, coeff in self.terms.items()
        ])

    def __add__(self, other):
        result = self.terms.copy()
        for op, coeff in other.terms.items():
            if op in result:
                result[op] += coeff
            else:
                result[op] = coeff
            if result[op] == 0:
                del result[op]
        return OperatorSum(result)
    
    def __sub__(self, other):
        result = self.terms.copy()
        for op, coeff in other.terms.items():
            if op in result:
                result[op] -= coeff
            else:
                result[op] = -coeff
            if result[op] == 0:
                del result[op]
        return OperatorSum(result)

    def __neg__(self):
        return OperatorSum({
            op: -coeff for op, coeff in self.terms.items()
        })

    def __rmul__(self, scalar):
        return OperatorSum({
            op: scalar * coeff for op, coeff in self.terms.items()
        })

    def mul(self, other):
        result = {}
        for op1, coeff1 in self.terms.items():
            for op2, coeff2 in other.terms.items():
                op, phase = op1.mul(op2)
                coeff = coeff1 * coeff2 * phase
                if op in result:
                    result[op] += coeff
                else:
                    result[op] = coeff
                if result[op] == 0:
                    del result[op]
        return OperatorSum(result)

    def dag(self):
        return OperatorSum({
            op.dag(): coeff.conjugate()
            for op, coeff in self.terms.items()
        })

    def commutator(self, other):
        return self.mul(other) - other.mul(self)


@dataclass(slots=True)
class IsingParams:
    L: int = 8
    J: float = 1.
    h: float = 1.


def build_hamil(params: IsingParams):
    assert params.L >= 2
    x_op = Operator('X'+'I'*(params.L-1))
    zz_op = Operator('ZZ'+'I'*(params.L-2))
    hamil_terms = {}
    for shift in range(params.L):
        op = zz_op.translate(shift)
        if op in hamil_terms:
            hamil_terms[op] += -params.J
        else:
            hamil_terms[op] = -params.J
        op = x_op.translate(shift)
        if op in hamil_terms:
            hamil_terms[op] += -params.h
        else:
            hamil_terms[op] = -params.h
    return OperatorSum(hamil_terms)


class IsingCompiler:
    params: IsingParams
    basis_tir: list[Operator] # length-L operator basis containing only translation-invariant representatives (TIR)
    basis: list[Operator]     # length-L operator basis after expanding all translations

    moment_index: dict[int, int]
    moment_ops: list[Operator]
    moment_matrix: list[list[LinearExpr]]
    constraints: list[LinearExpr]
    constraints_rank: int

    hamil_op: OperatorSum  # full hamiltonian operator for commutator computations
    hamil_expr: LinearExpr # compiled hamiltonian expression

    def __init__(self, params: IsingParams):
        self.L = params.L
        self.J = params.J
        self.h = params.h
        self.hamil_op = build_hamil(params)

    def compile(self, basis_str: list[str]):
        '''
            basis_str: list of TIR operators with reduced length
        '''
        for op_str in basis_str:
            if len(op_str) > self.L:
                raise ValueError('basis_tir operator length exceeds system size L')
        self.basis_tir = [Operator(op_str+'I'*(self.L-len(op_str))) for op_str in basis_str]
        self.basis = []
        basis_seen = set()
        for op in self.basis_tir:
            for shift in range(self.L):
                op_shift = op.translate(shift)
                if op_shift not in basis_seen:
                    basis_seen.add(op_shift)
                    self.basis.append(op_shift)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construct moment variables (TIR)
        self.moment_index = {}
        self.moment_ops = []
        self.moment_matrix = []

        # moment matrix is hermitian by construction
        for op1 in self.basis:
            row = []
            for op2 in self.basis:
                op, phase = op1.dag().mul(op2)
                key = op.canon()
                if key not in self.moment_index:
                    self.moment_index[key] = len(self.moment_ops)
                    self.moment_ops.append(op)
                row.append(LinearExpr(terms={self.moment_index[key]: phase}, const=0))
            self.moment_matrix.append(row)
        
        # hamiltonian must be representable in the current moment variable space
        self.hamil_expr = self.compile_expr(self.hamil_op)
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construct constraints
        self.constraints = []
        self.constraints_rank = 0

        # normalization constraint, <I> == 1
        id_key = Operator('I'*self.L).canon()
        if id_key not in self.moment_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.constraints.append(LinearExpr(
            terms={self.moment_index[id_key]: 1.},
            const=-1.,
        ))

        # eigenstate constraints, <[H,O]> == 0
        for op in self.moment_ops:
            comm_op = self.hamil_op.commutator(OperatorSum({op: 1.}))
            expr = self.compile_expr(comm_op)
            if expr is None:
                continue
            if expr.terms or expr.const != 0:
                self.constraints.append(expr)
        
        # symmtries
        ...

        # prune redundant constraints
        ...

        # compute the rank of constraints
        if not self.constraints:
            self.constraints_rank = 0
        else:
            mat = np.zeros(
                (len(self.constraints), len(self.moment_ops)),
                dtype=np.complex128,
            )
            for row, expr in enumerate(self.constraints):
                for idx, coeff in expr.terms.items():
                    mat[row, idx] = complex(coeff)
            self.constraints_rank = int(np.linalg.matrix_rank(mat))

    def summary(self):
        return {
            'L': self.L,
            'basis_tir': len(self.basis_tir),
            'basis': len(self.basis),
            'moment_ops': len(self.moment_ops),
            'constraints': len(self.constraints),
            'constraints_rank': self.constraints_rank,
            'hamil_expr': self._get_expr_str(self.hamil_expr),
        }

    def compile_expr(self, op_sum: OperatorSum) -> LinearExpr:
        expr = {}
        for op, coeff in op_sum.terms.items():
            key = op.canon()
            if key not in self.moment_index:
                return None
            idx = self.moment_index[key]
            if idx in expr:
                expr[idx] += coeff
            else:
                expr[idx] = coeff
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def _get_expr_str(self, expr: LinearExpr) -> str:
        parts = [
            f'{coeff}*<{self.moment_ops[idx]}>'
            for idx, coeff in expr.terms.items()
        ]
        if expr.const != 0:
            parts.append(str(expr.const))
        return ' + '.join(parts)


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    op1 = Operator(op_str='IXYZZZII')
    op2 = Operator(op_str='ZZZIIIXY')
    print(op1.canon(), op2.canon())

    print(op1)
    print(op2)
    print(*op1.mul(op2))

    op_sum1 = OperatorSum({op1: 1., op2: 1.j})
    op_sum2 = OperatorSum({op1: 1., op2: 1.j})
    print(op_sum1)
    print(op_sum2)
    print(op_sum1.mul(op_sum2))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    compiler = IsingCompiler(params=IsingParams())
    compiler.compile(basis_str=['I', 'X', 'Y', 'Z', 'ZZ'])
    print(*compiler.basis_tir)
    print(*compiler.basis)

    print(len(compiler.moment_ops))
    print(*compiler.moment_ops)

    print(compiler.hamil_op)
    print(compiler._get_expr_str(compiler.hamil_expr))

    print(len(compiler.constraints))
    for expr in compiler.constraints:
        print(compiler._get_expr_str(expr))

    print(compiler.summary())

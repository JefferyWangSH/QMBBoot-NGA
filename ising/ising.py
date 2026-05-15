from dataclasses import dataclass
from functools import cached_property
import numpy as np
import scipy as sp

from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData


'''
    length-L transverse Ising chain (PBC)
'''

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


class IsingOperator:
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
        op = IsingOperator()
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
        op = IsingOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr, -coeff)
        return op

    def __rmul__(self, scalar):
        op = IsingOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr, scalar * coeff)
        return op

    def mul(self, other):
        assert self.L is None or other.L is None or self.L == other.L
        op = IsingOperator()
        for pstr1, coeff1 in self.terms.items():
            for pstr2, coeff2 in other.terms.items():
                pstr, phase = pstr1.mul(pstr2)
                coeff = coeff1 * coeff2 * phase
                op.add(pstr, coeff)
        return op

    def dag(self):
        op = IsingOperator()
        for pstr, coeff in self.terms.items():
            op.add(pstr.dag(), coeff.conjugate())
        return op

    def commutator(self, other):
        return self.mul(other) - other.mul(self)


@dataclass(slots=True)
class IsingParams:
    L: int = 8
    J: float = 1.
    h: float = 1.


def build_hamil(params: IsingParams):
    assert params.L >= 2
    x = PauliString.from_str('X'+'I'*(params.L-1))
    zz = PauliString.from_str('ZZ'+'I'*(params.L-2))
    hamil_op = IsingOperator()
    for shift in range(params.L):
        hamil_op.add(zz.translate(shift), -params.J / params.L)
        hamil_op.add(x.translate(shift), -params.h / params.L)
    return hamil_op


class IsingCompiler:
    L: int
    params: IsingParams

    '''
        basis_reprs: translation-invariant representative Pauli strings with length L
    '''
    basis_reprs: list[PauliString]

    r'''
        moment matrix

            M_{ar,b0} = \langle O^\dag_a(r) O_b(0) \rangle
                       ~ \bigoplus_k M(k)_{ab}
        
        each O_a is a Pauli string in basis_reprs
        while the moment O^\dag_a(r) O_b(0) may acquire additional phase.
        note we use the associated hermitian Pauli string (excluding the phase) as the moment variables,
        whose expectation values parameterize the optimization space of bootstrap.

        K symmetry is used in
            1) reducing moment variables,
            2) relating M(k) and M(-k),
            3) and reducing stationarity constraint generators.

        var_cpx:      False
        vars:         K-even moment Pauli strings as SDP variables
        var_index:    map between canonical PauliString indices and variable indices
        ward_moments: K-odd moment Pauli strings for generating Ward identities
        ward_index:   map between canonical PauliString indices and Ward moment indices
        block_reprs:  representative basis involved in each momentum PSD block
        psd_blocks:   momentum PSD blocks
        affines:      affine constraints
        affines_mat:  affine constraints in terms of a sparse matrix
    '''
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[PauliString]

    ward_index: dict[int, int]
    ward_moments: list[PauliString]

    block_reprs: list[list[PauliString]]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: IsingOperator # full hamiltonian operator for computations of stationarity constraints
    hamil_expr: LinearExpr  # compiled hamiltonian expression

    def __init__(self, params: IsingParams):
        self.L = params.L
        self.params = params
        self.hamil_op = build_hamil(params)

    def _build_moments(self):
        '''
            build SDP variables and Ward moments

            moment variables self.vars involve only K-even Pauli strings because:

                1) the expectation value of K-odd Pauli string, which involves odd number of Y,
                   is purely imaginary given a K-symmetric denisty matrix.

                2) any Pauli string is hermitian so its expectation should be real.

            combining these facts yields <O_odd> = 0.
            therefore any K-odd Pauli string can be removed from moment variables.

            Pauli strings O in self.ward_moments are used to generate Ward identities, <[C,O]> == 0.
            as shown in note.ipynb, only K-odd Pauli strings generate nontrivial constraints.
        '''
        self.vars = []
        self.var_index = {}
        self.ward_moments = []
        self.ward_index = {}

        for pstr1 in self.basis_reprs:
            for r in range(pstr1.period):
                pstr1r = pstr1.translate(r)
                for pstr2 in self.basis_reprs:
                    pstr, _ = pstr1r.dag().mul(pstr2)
                    key = pstr.canon

                    if pstr.parity() == 1:
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(pstr)
                    else:
                        if key not in self.ward_index:
                            self.ward_index[key] = len(self.ward_moments)
                            self.ward_moments.append(pstr)

    @staticmethod
    def nonzero_fourier(pstr: PauliString, n: int) -> bool:
        return (n * pstr.period) % pstr.L == 0

    def _build_block_reprs(self):
        '''
            for a pauli string with period L_a < L,
            allowed momentum satisfy e^{-i k L_a} = 1 such that n L_a/L = m
        '''
        self.block_reprs = []
        for n in range(self.L//2 + 1):
            self.block_reprs.append([
                pstr for pstr in self.basis_reprs
                if self.nonzero_fourier(pstr, n)
            ])

    def _build_psd(self):
        '''
            build momentum PSD blocks M(k)
        '''
        self.psd_blocks = []

        r'''
            K symmetry imposes that M(-k) = diag(eta) M(k)^\ast diag(eta)
            therefore the number of independent momentum PSD blocks can be reduced by half
        '''
        for n, block_basis in enumerate(self.block_reprs):
            k = 2*np.pi * n / self.L
            psd = PSDConstraints(n_vars=len(self.vars), dim=len(block_basis))

            for row, pstr1 in enumerate(block_basis):
                for col, pstr2 in enumerate(block_basis):
                    expr = {}
                    for r in range(self.L):
                        pstr1r = pstr1.translate(r)
                        pstr, phase = pstr1r.dag().mul(pstr2)
                        if pstr.parity() == -1:
                            continue

                        idx = self.var_index[pstr.canon]
                        coeff = np.exp(1j * k * r) * phase / self.L
                        expr[idx] = expr.get(idx, 0) + coeff
                        if abs(expr[idx]) < 1e-12:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))

            self.psd_blocks.append(psd)

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        # normalization constraint, <I> == 1
        id_key = PauliString.from_str('I'*self.L).canon
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        for pstr in self.ward_moments:
            comm_op = self.hamil_op.commutator(IsingOperator({pstr: 1}))
            expr = self._compile_expr(comm_op)
            if expr is None:
                continue
            self.affines.add(expr)

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-10)

    def _compile_expr(self, op: IsingOperator) -> LinearExpr | None:
        expr = {}
        for pstr, coeff in op.terms.items():
            key = pstr.canon
            if key not in self.var_index:
                if pstr.parity() == -1:
                    continue
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def compile(self, basis_reprs: list[PauliString]):
        self.basis_reprs = list(basis_reprs)
        for pstr in self.basis_reprs:
            if pstr.L != self.L:
                raise ValueError('Pauli string length inconsistent with system size L')

        self._build_moments()
        self._build_block_reprs()
        self._build_psd()
        
        self.hamil_expr = self._compile_expr(self.hamil_op)
        # hamiltonian must be representable in the current moment variable space
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')
        
        self._build_affines()

    def local_comm(self, pstr: PauliString):
        r'''
            calculate

                C_a = [H, O_a(0)] = \sum_{b,s} C_{ab}(s) T_s O'(0)_b

            as entry list [(O'(0)_b, s, C_{ab}(s)), ...]
        '''
        entries = {}
        local_comm = self.hamil_op.commutator(IsingOperator({pstr: 1}))

        for desc, coeff in local_comm.terms.items():
            desc_rep = desc.canon_rep
            s = 0
            for shift in range(desc.L):
                if desc_rep.translate(shift) == desc:
                    s = shift
                    break

            key = (desc_rep, s)
            entries[key] = entries.get(key, 0) + coeff
            if entries[key] == 0:
                del entries[key]

        return [
            (desc_rep, s, coeff)
            for (desc_rep, s), coeff in entries.items()
        ]

    def _get_expr_str(self, expr: LinearExpr) -> str:
        parts = [
            f'{coeff}*<{str(self.vars[idx])}>'
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


def build_basis_reprs(L: int, basis: list[str]) -> list[PauliString]:
    basis_reprs = []
    for pstr in basis:
        if len(pstr) > L:
            raise ValueError('Pauli string length exceeds system size L')
        basis_reprs.append(PauliString.from_str(pstr + 'I'*(L - len(pstr))))
    return basis_reprs


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    pstr1 = PauliString.from_str(pstr='IXYZZZII')
    pstr2 = PauliString.from_str(pstr='ZZZIIIXY')
    print(pstr1.canon, pstr2.canon)

    print(pstr1)
    print(pstr2)
    print(*pstr1.mul(pstr2))

    op1 = IsingOperator({pstr1: 1., pstr2: 1.j})
    op2 = IsingOperator({pstr1: 1., pstr2: 1.j})
    print(op1)
    print(op2)
    print(op1.mul(op2))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    compiler = IsingCompiler(params=IsingParams())
    compiler.compile(basis_reprs=build_basis_reprs(compiler.L, ['I', 'X', 'Y', 'Z', 'ZZ']))
    print(*compiler.basis_reprs)
    print(len(compiler.vars))
    print(*compiler.vars)

    print(compiler.hamil_op)
    print(compiler._get_expr_str(compiler.hamil_expr))
    print(compiler.summary())

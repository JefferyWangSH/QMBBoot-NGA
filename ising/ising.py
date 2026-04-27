from dataclasses import dataclass
import scipy as sp

from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData


'''
    length-L transverse Ising chain (PBC)
'''

class PauliString:
    L: int      # bit length
    x_mask: int # bit representation of x operators
    z_mask: int # bit representation of z operators

    # ocanonical representation, unique as the translation-invariant representative
    _canon: int | None

    def __init__(self, pstr: str):
        self.L = len(pstr)
        self.x_mask = 0
        self.z_mask = 0
        for i, pauli in enumerate(pstr):
            if pauli in 'XY':
                self.x_mask |= 1 << i
            if pauli in 'ZY':
                self.z_mask |= 1 << i
        self._canon = None

    @classmethod
    def _from_masks(cls, L: int, x_mask: int, z_mask: int):
        pstr = cls.__new__(cls)
        pstr.L = L
        pstr.x_mask = x_mask
        pstr.z_mask = z_mask
        pstr._canon = None
        return pstr
    
    def _rotate_l(self, mask: int, shift: int) -> int:
        shift %= self.L
        if shift == 0:
            return mask
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
        pstr = PauliString.__new__(PauliString)
        pstr.L = self.L
        pstr.x_mask = self._rotate_l(self.x_mask, shift)
        pstr.z_mask = self._rotate_l(self.z_mask, shift)
        pstr._canon = self._canon
        return pstr

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

    def mul(self, other) -> tuple["PauliString", float|complex]:
        assert self.L == other.L
        x_mask = self.x_mask ^ other.x_mask
        z_mask = self.z_mask ^ other.z_mask

        phase = 1.
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

        return PauliString._from_masks(self.L, x_mask, z_mask), phase

    def parity(self):
        # +1 for K even and -1 for K odd
        return 1 - 2 * int((self.x_mask & self.z_mask).bit_count() % 2)


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
    x = PauliString('X'+'I'*(params.L-1))
    zz = PauliString('ZZ'+'I'*(params.L-2))
    hamil_op = IsingOperator()
    for shift in range(params.L):
        hamil_op.add(zz.translate(shift), -params.J)
        hamil_op.add(x.translate(shift), -params.h)
    return hamil_op


class IsingCompiler:
    params: IsingParams

    '''
        basis_reprs:     translation-invariant representative Pauli strings with length L
        basis_full:      full length-L basis after expanding all translations
        basis_full_even: K-even subset of basis_full
        basis_full_odd:  K-odd subset of basis_full
    '''
    basis_reprs: list[PauliString]
    basis_full: list[PauliString]
    basis_full_even: list[PauliString]
    basis_full_odd: list[PauliString]

    r'''
        moment matrix

            M_{ij} = \langle O^\dag_i O_j \rangle
        
        each O_i is a Pauli string while the moment O^\dag_i O_j may acquire additional phase

        note we use the associated Pauli string (excluding the phase) as the moment variables,
        whose expectation values parameterize the optimization space of bootstrap

        K symmetry is used in
            1) reducing moment variables,
            2) reducing stationarity constraint generators,
            3) and transforming M into a real symmetric matrix.

        var_cpx:      False
        var_index:    map between canonical PauliString indices and variable indices
        vars:         moment Pauli strings as optimization variables (vars = moments_even)
        moments_even: K-even moment Pauli strings (translation-invariant representatives)
        moments_odd:  K-odd moment Pauli strings (translation-invariant representatives)
        psd_blocks:   moment PSD blocks
        affines:      affine constraints
        affines_mat:  affine constraints in terms of a sparse matrix
    '''
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[PauliString]
    moments_even: list[PauliString]
    moments_odd: list[PauliString]

    psd_blocks: list[PSDConstraints]
    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: IsingOperator # full hamiltonian operator for computations of stationarity constraints
    hamil_expr: LinearExpr  # compiled hamiltonian expression

    def __init__(self, params: IsingParams):
        self.L = params.L
        self.J = params.J
        self.h = params.h
        self.hamil_op = build_hamil(params)

    def _build_moments(self):
        '''
            the density matrix is assumed real and symmetric
            without loss of generality for Ising model given the anti-unitary K symmetry

            moment variables self.vars involve only K-even Pauli strings because:

                1) the expectation value of K-odd Pauli string, which involves odd number of Y,
                   is purely imaginary given a symmetric denisty matrix.
                
                2) any Pauli string is hermitian so its expectation should be real.

            combining these facts yields <O_odd> = 0.
            therefore any K-odd Pauli string can be removed from moment variables.
        '''
        self.var_index = {}
        self.vars = []
        self.moments_even = []
        self.moments_odd = []

        moments_even_index = {}
        moments_odd_index = {}

        for pstr1 in self.basis_full:
            for pstr2 in self.basis_full:
                pstr, phase = pstr1.dag().mul(pstr2)
                key = pstr.canon()

                if pstr.parity() == 1:
                    if key not in moments_even_index:
                        moments_even_index[key] = len(self.moments_even)
                        self.moments_even.append(pstr)
                else:
                    if key not in moments_odd_index:
                        moments_odd_index[key] = len(self.moments_odd)
                        self.moments_odd.append(pstr)

        self.var_index = moments_even_index
        self.vars = self.moments_even

        # # plain benchmark: construct moment psd matrix as a hermitian
        # psd = PSDConstraints(n_vars=len(self.vars), dim=len(self.basis_full))
        # for i, pstr1 in enumerate(self.basis_full):
        #     for j, pstr2 in enumerate(self.basis_full):
        #         pstr, phase = pstr1.dag().mul(pstr2)
        #         if pstr.parity() == 1:
        #             psd.add(i, j, LinearExpr(terms={self.var_index[pstr.canon()]: phase}, const=0))
        # self.psd_blocks = [psd]

        # transform moment psd matrix to a real symmetric matrix
        psd = PSDConstraints(n_vars=len(self.vars), dim=len(self.basis_full))
        n_even = len(self.basis_full_even)
        for i, pstr1 in enumerate(self.basis_full):
            row_even = i < n_even

            for j, pstr2 in enumerate(self.basis_full):
                col_even = j < n_even
                pstr, phase = pstr1.dag().mul(pstr2)
                key = pstr.canon()

                if pstr.parity() == -1:
                    continue

                # M = [[M_1, i M_3], [-i M_3^T, M_2]]
                # U = diag(I, iI)
                # \tilde{M} = U^\dag M U = [[M_1, -M_3], [-M_3^T, M_2]]
                factor_i = 1. if row_even else -1j
                factor_j = 1. if col_even else 1j
                coeff = factor_i * phase * factor_j

                # after transformation every entry should be real
                coeff = complex(coeff)
                if abs(coeff.imag) > 1e-12:
                    raise ValueError(
                        f'unexpected imaginary entry in transformed M: {coeff} '
                        f'from <{str(pstr1.dag())} * {str(pstr2)}> -> {str(pstr)}'
                    )
                psd.add(i, j, LinearExpr(
                    terms={self.var_index[key]: float(coeff.real)},
                    const=0,
                ))
        self.psd_blocks = [psd]

    def _compile_expr(self, op: IsingOperator) -> LinearExpr | None:
        expr = {}
        for pstr, coeff in op.terms.items():
            key = pstr.canon()
            if key not in self.var_index:
                if pstr.parity() == -1:
                    continue
                return None
            idx = self.var_index[key]
            expr[idx] = expr.get(idx, 0) + coeff
            if expr[idx] == 0:
                del expr[idx]
        return LinearExpr(terms=expr, const=0)

    def compile(self, basis: list[str]):
        '''
            basis: list of translation-invariant representative Pauli strings with reduced length
        '''
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ build basis
        for pstr in basis:
            if len(pstr) > self.L:
                raise ValueError('Pauli string length exceeds system size L')

        self.basis_reprs = [PauliString(pstr+'I'*(self.L-len(pstr))) for pstr in basis]
        self.basis_full_even = []
        self.basis_full_odd = []
        basis_seen = set()
        for pstr in self.basis_reprs:
            is_even = pstr.parity() == 1
            for shift in range(self.L):
                pstr_shift = pstr.translate(shift)
                if pstr_shift not in basis_seen:
                    basis_seen.add(pstr_shift)
                    if is_even:
                        self.basis_full_even.append(pstr_shift)
                    else:
                        self.basis_full_odd.append(pstr_shift)
        # order basis so that M has the K-even / K-odd block structure
        self.basis_full = self.basis_full_even + self.basis_full_odd

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construct moment related
        self._build_moments()
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construct hamiltonian
        self.hamil_expr = self._compile_expr(self.hamil_op)
        # hamiltonian must be representable in the current moment variable space
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ construct affine constraints
        self.affines = AffineConstraints(n_vars=len(self.vars))

        # normalization constraint, <I> == 1
        id_key = PauliString('I'*self.L).canon()
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        # stationarity constraints, <[H,O]> == 0, for Pauli strings O in self.moments_odd
        # since moment variables are real for sure, the constraints have only real coefficients.
        # as shown in note.ipynb, only K-odd Pauli strings generate nontrivial constraints.
        for pstr in self.moments_odd:
            comm_op = self.hamil_op.commutator(IsingOperator({pstr: 1}))
            expr = self._compile_expr(comm_op)
            if expr is None:
                continue
            self.affines.add(expr)

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-10)

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
            'L': self.L,
            'basis_reprs': len(self.basis_reprs),
            'basis_full': len(self.basis_full),
            'moments': {
                'even': len(self.moments_even),
                'odd': len(self.moments_odd),
                'total': len(self.moments_even)+len(self.moments_odd),
            },
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


if __name__ == '__main__':

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    pstr1 = PauliString(pstr='IXYZZZII')
    pstr2 = PauliString(pstr='ZZZIIIXY')
    print(pstr1.canon(), pstr2.canon())

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
    compiler.compile(basis=['I', 'X', 'Y', 'Z', 'ZZ'])
    print(*compiler.basis_reprs)
    print(*compiler.basis_full)

    print(len(compiler.vars))
    print(*compiler.vars)

    print(compiler.hamil_op)
    print(compiler._get_expr_str(compiler.hamil_expr))
    print(compiler.summary())

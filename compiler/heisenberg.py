from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from operators.pauli import PauliString, PauliOperator
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData


@dataclass(slots=True)
class HeisenbergParams:
    L: int = 8
    J1: float = 1.
    J2: float = 1.


def build_hamil(params: HeisenbergParams):
    assert params.L >= 3
    hamil_op = PauliOperator()

    def _two_site_op(L: int, pauli: str, dist: int):
        return PauliString.from_str(pauli + 'I'*(dist-1) + pauli + 'I'*(L-dist-1))

    for pauli in 'XYZ':
        nn = _two_site_op(params.L, pauli, 1)
        nnn = _two_site_op(params.L, pauli, 2)
        for shift in range(params.L):
            hamil_op.add(nn.translate(shift), .25 * params.J1 / params.L)
            hamil_op.add(nnn.translate(shift), .25 * params.J2 / params.L)

    return hamil_op


class HeisenbergCompiler:
    L: int
    params: HeisenbergParams

    basis_reprs: list[PauliString]

    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[PauliString]

    ward_index: dict[str, dict[int, int]]
    ward_moments: dict[str, list[PauliString]]

    _moment_var = 1 << 0
    _moment_ward = 1 << 1
    _moment_flags_cache: dict[int, int]
    _sym_canon_cache: dict[int, int]

    # we divide PSD blocks through both translation and spin-rotation of pi angle
    block_reprs: list[list[PauliString]]
    block_momenta: list[int]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: PauliOperator
    hamil_expr: LinearExpr

    def __init__(self, params: HeisenbergParams):
        self.L = params.L
        self.params = params
        self.hamil_op = build_hamil(params)
        self._moment_flags_cache = {}
        self._sym_canon_cache = {}

    def _flag(self, pstr: PauliString):
        if pstr.mask in self._moment_flags_cache:
            return self._moment_flags_cache[pstr.mask]

        '''
            1) vars must have sign charge 000, which is definitely also K-even.

            2) [H,O] preserves pi-rotation charge, therefore to generate non-trivial stationarity constraints
               O must have pi-rotation charge ++, i.e. sign charge 000/111.
               Among them, only 111 sector is K-odd and gives K-even commutators.

            3) non-trivial SO(3) Ward constraints come from commutations with charges S^a_{tot} (100, 010 and 001),
               hence ward moments must have sign charges 011, 101, or 110.

            In summary,

                000:         SDP variables
                111:         Hamiltonian Ward operators
                011/101/110: SO(3) Ward operators
        '''
        charge = pstr.sign_charge()
        flag = 0
        if charge == (0, 0, 0):
            flag |= self._moment_var
        elif charge in ((1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
            flag |= self._moment_ward

        self._moment_flags_cache[pstr.mask] = flag
        return flag

    def _is_var(self, pstr: PauliString):
        return bool(self._flag(pstr) & self._moment_var) # 000

    def _is_ward(self, pstr: PauliString):
        return bool(self._flag(pstr) & self._moment_ward) # 111/011/101/110

    def _is_zero(self, pstr: PauliString):
        return not self._is_var(pstr) # not 000

    def _sym_canon(self, pstr: PauliString):
        if pstr.mask in self._sym_canon_cache:
            return self._sym_canon_cache[pstr.mask]

        orbit = [pstr.permute(perm) for perm in itertools.permutations('XYZ')]
        key = min(rep.canon for rep in orbit)
        for rep in orbit:
            self._sym_canon_cache[rep.mask] = key
        return key

    def _build_moments(self):
        self.vars = []
        self.var_index = {}
        self.ward_moments = {'hamil': [], 'spin': []}
        self.ward_index = {'hamil': {}, 'spin': {}}

        for pstr1 in self.basis_reprs:
            for r in range(pstr1.period):
                pstr1r = pstr1.translate(r)
                for pstr2 in self.basis_reprs:
                    pstr, _ = pstr1r.dag().mul(pstr2)

                    if self._is_var(pstr):
                        key = self._sym_canon(pstr)
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(PauliString(self.L, key))

                    if self._is_ward(pstr):
                        charge = pstr.sign_charge()
                        kind = 'hamil' if charge == (1, 1, 1) else 'spin'
                        key = pstr.canon
                        if key not in self.ward_index[kind]:
                            self.ward_index[kind][key] = len(self.ward_moments[kind])
                            self.ward_moments[kind].append(PauliString(self.L, key))

    @staticmethod
    def nonzero_fourier(pstr: PauliString, n: int) -> bool:
        return (n * pstr.period) % pstr.L == 0

    def _build_block_reprs(self):
        self.block_reprs = []
        self.block_momenta = []

        for n in range(self.L//2 + 1): # K symmetry
            charge_reprs = {}
            for pstr in self.basis_reprs:
                if not self.nonzero_fourier(pstr, n):
                    continue
                charge = pstr.pi_rot_charge()
                charge_reprs.setdefault(charge, []).append(pstr)

            for charge in sorted(charge_reprs):
                self.block_reprs.append(charge_reprs[charge])
                self.block_momenta.append(n)

    def _build_psd(self):
        self.psd_blocks = []

        for n, block_basis in zip(self.block_momenta, self.block_reprs):
            k = 2*np.pi * n / self.L
            psd = PSDConstraints(n_vars=len(self.vars), dim=len(block_basis))

            for row, pstr1 in enumerate(block_basis):
                for col, pstr2 in enumerate(block_basis):
                    expr = {}
                    for r in range(self.L):
                        pstr1r = pstr1.translate(r)
                        pstr, phase = pstr1r.dag().mul(pstr2)
                        if self._is_zero(pstr):
                            continue

                        idx = self.var_index[self._sym_canon(pstr)]
                        coeff = np.exp(1j * k * r) * phase / self.L
                        expr[idx] = expr.get(idx, 0) + coeff
                        if abs(expr[idx]) < 1e-12:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))

            self.psd_blocks.append(psd)

    def _total_spin_comm(self, pstr: PauliString, axis_code: int) -> PauliOperator:
        '''
            return [S^a_tot, O] as a Pauli operator
            axis_code: X=1=01, Z=2=10, Y=3=11
        '''
        X, Y, Z = 1, 3, 2
        coeffs = {
            (X, Y): 1j,  # [Sx, Y] =  i Z
            (Y, Z): 1j,  # [Sy, Z] =  i X
            (Z, X): 1j,  # [Sz, X] =  i Y
            (X, Z): -1j, # [Sx, Z] = -i Y
            (Z, Y): -1j, # [Sz, Y] = -i X
            (Y, X): -1j, # [Sy, X] = -i Z
        }
        op = PauliOperator()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            old_code = (pstr.mask >> (2*site)) & 3
            support &= ~(3 << (2*site))
            if old_code == axis_code:
                continue

            new_mask = (pstr.mask & ~(3 << (2*site))) | ((axis_code ^ old_code) << (2*site))
            op.add(PauliString(self.L, new_mask), coeffs[(axis_code, old_code)])
        return op

    def _hamil_comm(self, pstr: PauliString) -> PauliOperator:
        if not hasattr(self, '_hamil_terms_by_site'):
            self._hamil_terms_by_site = [[] for _ in range(self.L)]
            for hstr, hcoeff in self.hamil_op.terms.items():
                support = hstr.mask
                while support:
                    bit = support & -support
                    site = (bit.bit_length() - 1) // 2
                    support &= ~(3 << (2*site))
                    self._hamil_terms_by_site[site].append((hstr, hcoeff))

        op = PauliOperator()
        seen = set()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            support &= ~(3 << (2*site))

            for hstr, hcoeff in self._hamil_terms_by_site[site]:
                if hstr.mask in seen:
                    continue
                seen.add(hstr.mask)

                anticomm = 0
                h_support = hstr.mask
                while h_support:
                    h_bit = h_support & -h_support
                    h_site = (h_bit.bit_length() - 1) // 2
                    h_support &= ~(3 << (2*h_site))

                    h_code = (hstr.mask >> (2*h_site)) & 3
                    p_code = (pstr.mask >> (2*h_site)) & 3
                    if p_code != 0 and p_code != h_code:
                        anticomm ^= 1

                if anticomm:
                    prod, phase = hstr.mul(pstr)
                    op.add(prod, 2 * hcoeff * phase)
        return op

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))

        id_key = self._sym_canon(PauliString.from_str('I'*self.L))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        # stationarity constraints
        for pstr in self.ward_moments['hamil']:
            expr = self._compile_expr(self._hamil_comm(pstr))
            if expr is None:
                continue
            self.affines.add(expr)

        # SO(3) Ward identities
        for axis_code in (1, 3, 2): # X, Y, Z
            for pstr in self.ward_moments['spin']:
                expr = self._compile_expr(self._total_spin_comm(pstr, axis_code))
                if expr is None:
                    continue
                self.affines.add(expr)

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-10)

    def _compile_expr(self, op: PauliOperator) -> LinearExpr | None:
        expr = {}
        for pstr, coeff in op.terms.items():
            if self._is_zero(pstr):
                continue
            key = self._sym_canon(pstr)
            if key not in self.var_index:
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
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        self._build_affines()

    def descendants(self, pstr: PauliString):
        entries = {}
        comm = self._hamil_comm(pstr)

        for desc, coeff in comm.terms.items():
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
            'ward_moments': {kind: len(moments) for kind, moments in self.ward_moments.items()},
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

    params = HeisenbergParams(L=8, J1=1., J2=1.)
    compiler = HeisenbergCompiler(params)
    compiler.compile(build_basis_reprs(params.L, ['I', 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XXX', 'YYY', 'ZZZ']))

    print(*compiler.basis_reprs)
    print(compiler.summary())

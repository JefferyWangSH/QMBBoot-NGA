from dataclasses import dataclass
from lru import LRU
import numpy as np
import scipy as sp

from operators.pauli import PauliString, PauliOperator, _PAULI_PHASE, _PAULI_MUL_PHASE_POWER
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData

_CACHE_MAX_SIZE = 10_000_000


'''
    length-L transverse/longitudinal-field Ising chain (PBC)
'''

@dataclass(slots=True)
class IsingParams:
    L: int = 8
    J: float = 1.
    h: float = 1.
    hz: float = 0.


def build_hamil(params: IsingParams):
    assert params.L >= 2
    x = PauliString.from_str('X'+'I'*(params.L-1))
    zz = PauliString.from_str('ZZ'+'I'*(params.L-2))
    hamil_op = PauliOperator()
    for shift in range(params.L):
        hamil_op.add(zz.translate(shift), -params.J / params.L)
        hamil_op.add(x.translate(shift), -params.h / params.L)

    if params.hz != 0:
        z = PauliString.from_str('Z'+'I'*(params.L-1))
        for shift in range(params.L):
            hamil_op.add(z.translate(shift), -params.hz / params.L)

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

        K symmetry is used in 1) reducing moment variables and 2) relating M(k) and M(-k).

        var_cpx:       False
        vars:          K-even moment Pauli strings as SDP variables
        var_index:     map between canonical PauliString indices and variable indices
        ward_ops:      number of Ward-identity operators O used to generate constraints
        block_reprs:   representative basis involved in each momentum PSD block
        block_momenta: momentum index to which each PSD block corresponds
        psd_blocks:    momentum PSD blocks
        affines:       affine constraints
        affines_mat:   affine constraints in terms of a sparse matrix
    '''
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[PauliString]
    ward_ops: dict[str, int]

    block_reprs: list[list[PauliString]]
    block_momenta: list[int]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: PauliOperator # full hamiltonian operator for computations of stationarity constraints
    hamil_expr: LinearExpr  # compiled hamiltonian expression

    # Hamiltonian terms bucketed by every site in their support
    # for each site, it involves a list of
    # h_sites: tuple of (site, Pauli code)
    # h_mask: Hamiltonian Pauli mask
    # h_coeff: Hamiltonian coefficient
    _hamil_terms_at_site: list[list[tuple[tuple[tuple[int, int], ...], int, float | complex]]]

    def __init__(self, params: IsingParams):
        self.L = params.L
        self.params = params
        self.ward_ops = {'hamil': 0}

        self.hamil_op = build_hamil(params)
        self._hamil_terms_at_site = [[] for _ in range(self.L)]
        for hstr, h_coeff in self.hamil_op.terms.items():
            h_mask = hstr.mask
            h_sites = []
            support = h_mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 2
                code = (h_mask >> (2*site)) & 3
                h_sites.append((site, code))
                support &= ~(3 << (2*site))

            h_sites = tuple(h_sites)
            for site, _ in h_sites:
                self._hamil_terms_at_site[site].append((h_sites, h_mask, h_coeff))

    def trans_canon(self, pstr: PauliString) -> int:
        if not hasattr(self, '_trans_canon_cache'):
            self._trans_canon_cache = LRU(_CACHE_MAX_SIZE)
        if pstr.mask in self._trans_canon_cache:
            return self._trans_canon_cache[pstr.mask]

        key = PauliString(self.L, pstr.mask).trans_canon
        self._trans_canon_cache[pstr.mask] = key
        return key

    def trans_canon_rep(self, pstr: PauliString) -> PauliString:
        return PauliString(L=self.L, mask=self.trans_canon(pstr))

    def _sym_canon(self, pstr: PauliString) -> int:
        if not hasattr(self, '_sym_canon_cache'):
            self._sym_canon_cache = LRU(_CACHE_MAX_SIZE)
        if pstr.mask in self._sym_canon_cache:
            return self._sym_canon_cache[pstr.mask]

        key = min(pstr.trans_canon, pstr.invert().trans_canon)
        self._sym_canon_cache[pstr.mask] = key
        return key

    def _sym_allowed(self, pstr: PauliString) -> bool:
        if not hasattr(self, '_sym_allowed_cache'):
            self._sym_allowed_cache = LRU(_CACHE_MAX_SIZE)
        if pstr.mask in self._sym_allowed_cache:
            return self._sym_allowed_cache[pstr.mask]

        allowed = PauliString(self.L, pstr.mask).parity() == 0
        self._sym_allowed_cache[pstr.mask] = allowed
        return allowed

    def _build_vars(self):
        '''
            build K-even SDP variables

            SDP variables involve only K-even Pauli strings because:

                1) the expectation value of K-odd Pauli string, which involves odd number of Y,
                   is purely imaginary given a K-symmetric denisty matrix.

                2) any Pauli string is hermitian so its expectation should be real.

            combining these facts yields <O_odd> = 0,
            therefore any K-odd Pauli string can be removed from SDP variables.
        '''
        self.vars = []
        self.var_index = {}

        for pstr1 in self.basis_reprs:
            for r in range(pstr1.period):
                pstr1r = pstr1.translate(r)
                for pstr2 in self.basis_reprs:
                    pstr, _ = pstr1r.dag().mul(pstr2)

                    if self._sym_allowed(pstr):
                        key = self._sym_canon(pstr)
                        if key not in self.var_index:
                            self.var_index[key] = len(self.vars)
                            self.vars.append(PauliString(L=self.L, mask=key))

    @staticmethod
    def nonzero_fourier(pstr: PauliString, n: int) -> bool:
        return (n * pstr.period) % pstr.L == 0

    def _build_block_reprs(self):
        '''
            for a pauli string with period L_a < L,
            allowed momentum satisfy e^{-i k L_a} = 1 such that n L_a/L = m
        '''
        self.block_reprs = []
        self.block_momenta = []
        for n in range(self.L//2 + 1):
            self.block_reprs.append([
                pstr for pstr in self.basis_reprs
                if self.nonzero_fourier(pstr, n)
            ])
            self.block_momenta.append(n)

    def _build_psd(self):
        '''
            build momentum PSD blocks M(k)
        '''
        self.psd_blocks = []

        r'''
            K symmetry imposes that M(-k) = diag(eta) M(k)^\ast diag(eta)
            therefore the number of independent momentum PSD blocks can be reduced by half
        '''
        for n, block_basis in zip(self.block_momenta, self.block_reprs):
            k = 2*np.pi * n / self.L
            psd = PSDConstraints(n_vars=len(self.vars), dim=len(block_basis))

            for row, pstr1 in enumerate(block_basis):
                for col, pstr2 in enumerate(block_basis):
                    expr = {}
                    for r in range(self.L):
                        pstr1r = pstr1.translate(r)
                        pstr, phase = pstr1r.dag().mul(pstr2)
                        if not self._sym_allowed(pstr):
                            continue

                        idx = self.var_index[self._sym_canon(pstr)]
                        coeff = np.exp(1j * k * r) * phase / self.L
                        expr[idx] = expr.get(idx, 0) + coeff
                        if abs(expr[idx]) < 1e-14:
                            del expr[idx]
                    psd.add(row, col, LinearExpr(terms=expr, const=0))

            self.psd_blocks.append(psd)

    def _hamil_comm(self, pstr: PauliString) -> PauliOperator:
        op = PauliOperator()
        seen = set()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            support &= ~(3 << (2*site))

            for h_sites, h_mask, h_coeff in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                anticomm = 0
                phase_power = 0
                for h_site, h_code in h_sites:
                    p_code = (pstr.mask >> (2*h_site)) & 3
                    if p_code != 0 and p_code != h_code:
                        anticomm ^= 1
                    phase_power += _PAULI_MUL_PHASE_POWER[h_code][p_code]

                if anticomm:
                    prod = PauliString(self.L, h_mask ^ pstr.mask)
                    op.add(prod, 2 * h_coeff * _PAULI_PHASE[phase_power & 3])
        return op

    def _compile_hamil_ward(self, pstr: PauliString) -> LinearExpr | None:
        expr = {}
        seen = set()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            support &= ~(3 << (2*site))

            for h_sites, h_mask, h_coeff in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                anticomm = 0
                phase_power = 0
                for h_site, h_code in h_sites:
                    p_code = (pstr.mask >> (2*h_site)) & 3
                    if p_code != 0 and p_code != h_code:
                        anticomm ^= 1
                    phase_power += _PAULI_MUL_PHASE_POWER[h_code][p_code]

                # given K-odd pstr and K-even hstr, hstr*pstr carries one net i phase if they anticommute.
                # so the resulting hermitian Pauli string prod must be K-even.
                if not anticomm:
                    continue

                prod = PauliString(self.L, h_mask ^ pstr.mask)
                key = self._sym_canon(prod)
                if key not in self.var_index:
                    return None

                idx = self.var_index[key]
                expr[idx] = expr.get(idx, 0) + 2 * h_coeff * _PAULI_PHASE[phase_power & 3]
                if expr[idx] == 0:
                    del expr[idx]

        return LinearExpr(terms=expr, const=0)

    def _add_hamil_wards(self):
        r'''
            add all representable stationarity Ward identities <[H, O]> == 0
            within the current SDP variable set

            as explicitly ensured in this function,
            non-trivial stationarity constraints come from K-odd O.
        '''
        seen_masks = set()
        seen_keys = set()
        for pstr in self.vars:
            support = pstr.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 2
                support &= ~(3 << (2*site))

                for h_sites, h_mask, _ in self._hamil_terms_at_site[site]:
                    anticomm = 0
                    for h_site, h_code in h_sites:
                        p_code = (pstr.mask >> (2*h_site)) & 3
                        if p_code != 0 and p_code != h_code:
                            anticomm ^= 1

                    # since both SDP variable pstr and hstr are K-even,
                    # anticommutation makes the candidate mask, i.e. mask = pstr.mask ^ hstr.mask, K-odd.
                    if not anticomm:
                        continue

                    mask = pstr.mask ^ h_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = PauliString(self.L, mask)
                    # trans_canon is cheaper than _sym_canon
                    # although it may produce redundant Ward identities
                    key = self.trans_canon(cand)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_hamil_ward(cand)
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops['hamil'] += 1

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))
        self.ward_ops = {'hamil': 0}

        # normalization constraint, <I> == 1
        id_key = self._sym_canon(PauliString.from_str('I'*self.L))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        self._add_hamil_wards()

        self.affines_mat, _ = self.affines.matrix(prune=True, tol=1e-12)

    def _compile_expr(self, op: PauliOperator) -> LinearExpr | None:
        expr = {}
        for pstr, coeff in op.terms.items():
            if not self._sym_allowed(pstr):
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

        self._build_vars()
        self._build_block_reprs()
        self._build_psd()
        
        self.hamil_expr = self._compile_expr(self.hamil_op)
        # hamiltonian must be representable in the current moment variable space
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')
        
        self._build_affines()

    def descendants(self, pstr: PauliString):
        r'''
            calculate

                C_a = [H, O_a(0)] = \sum_{b,s} C_{ab}(s) T_s O'(0)_b

            as entry list [(O'(0)_b, s, C_{ab}(s)), ...]
        '''
        if not hasattr(self, '_descendants_cache'):
            self._descendants_cache = LRU(_CACHE_MAX_SIZE)
        if pstr.mask in self._descendants_cache:
            return self._descendants_cache[pstr.mask]

        entries = {}
        comm_op = self._hamil_comm(pstr)

        for desc, coeff in comm_op.terms.items():
            desc_rep = self.trans_canon_rep(desc)
            s = 0
            for shift in range(desc.L):
                if desc_rep.translate(shift) == desc:
                    s = shift
                    break

            key = (desc_rep.mask, s)
            entries[key] = entries.get(key, 0) + coeff
            if entries[key] == 0:
                del entries[key]

        descs = [
            (PauliString(self.L, desc_mask), s, coeff)
            for (desc_mask, s), coeff in entries.items()
        ]
        self._descendants_cache[pstr.mask] = descs
        return descs

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
            'ward_ops': self.ward_ops,
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
    print(pstr1.trans_canon, pstr2.trans_canon)

    print(pstr1)
    print(pstr2)
    print(*pstr1.mul(pstr2))

    op1 = PauliOperator({pstr1: 1., pstr2: 1.j})
    op2 = PauliOperator({pstr1: 1., pstr2: 1.j})
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

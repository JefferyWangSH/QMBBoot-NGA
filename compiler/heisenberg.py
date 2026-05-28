from dataclasses import dataclass
from lru import LRU
import itertools
import numpy as np
import scipy as sp

from operators.pauli import PauliString, PauliOperator, _PAULI_CODE, _PAULI_PHASE, _PAULI_MUL_PHASE_POWER
from sdp import LinearExpr, PSDConstraints, AffineConstraints, SDPData

_CACHE_MAX_SIZE = 10_000_000


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

    ward_ops: dict[str, int]

    # we divide PSD blocks through both translation and spin-rotation of pi angle
    block_reprs: list[list[PauliString]]
    block_momenta: list[int]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: PauliOperator
    hamil_expr: LinearExpr
    _hamil_terms_at_site: list[list[tuple[int, int, int, float | complex]]]

    def __init__(self, params: HeisenbergParams):
        self.L = params.L
        self.params = params
        self.ward_ops = {'hamil': 0, 'Sx': 0, 'Sy': 0, 'Sz': 0}

        self.hamil_op = build_hamil(params)
        self._hamil_terms_at_site = [[] for _ in range(self.L)]
        for hstr, h_coeff in self.hamil_op.terms.items():
            h_mask = hstr.mask
            sites = []
            support = h_mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 2
                code = (h_mask >> (2 * site)) & 3
                sites.append((site, code))
                support &= ~(3 << (2 * site))

            (site1, code1), (site2, code2) = sites
            self._hamil_terms_at_site[site1].append((site2, code1, h_mask, h_coeff))
            self._hamil_terms_at_site[site2].append((site1, code2, h_mask, h_coeff))

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

        orbit = []
        for base in (pstr, pstr.invert()):
            for perm in itertools.permutations('XYZ'):
                orbit.append(base.permute(perm))

        # permutations commute with translation, and inversion maps T_s to T_-s,
        # so taking trans_canon after each inversion/S3 image covers the full orbit.
        key = min(rep.trans_canon for rep in orbit)
        for rep in orbit:
            self._sym_canon_cache[rep.mask] = key
        return key

    def _sym_allowed(self, pstr: PauliString) -> bool:
        if not hasattr(self, '_sym_allowed_cache'):
            self._sym_allowed_cache = LRU(_CACHE_MAX_SIZE)
        if pstr.mask in self._sym_allowed_cache:
            return self._sym_allowed_cache[pstr.mask]

        allowed = PauliString(self.L, pstr.mask).sign_charge() == (0, 0, 0)
        self._sym_allowed_cache[pstr.mask] = allowed
        return allowed

    def _build_vars(self):
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
                            self.vars.append(PauliString(self.L, key))

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
            p_code = (pstr.mask >> (2 * site)) & 3
            support &= ~(3 << (2 * site))

            for neighbor_site, axis_code, h_mask, h_coeff in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                neighbor_code = (pstr.mask >> (2 * neighbor_site)) & 3
                anticomm = (
                    (p_code != 0 and p_code != axis_code)
                  ^ (neighbor_code != 0 and neighbor_code != axis_code)
                )
                if anticomm:
                    phase_power = (
                        _PAULI_MUL_PHASE_POWER[axis_code][p_code]
                      + _PAULI_MUL_PHASE_POWER[axis_code][neighbor_code]
                    )
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
            p_code = (pstr.mask >> (2 * site)) & 3
            support &= ~(3 << (2 * site))

            for neighbor_site, axis_code, h_mask, h_coeff in self._hamil_terms_at_site[site]:
                if h_mask in seen:
                    continue
                seen.add(h_mask)

                neighbor_code = (pstr.mask >> (2 * neighbor_site)) & 3
                anticomm = (
                    (p_code != 0 and p_code != axis_code)
                  ^ (neighbor_code != 0 and neighbor_code != axis_code)
                )
                if not anticomm:
                    continue

                prod = PauliString(self.L, h_mask ^ pstr.mask)
                prod_key = self._sym_canon(prod)
                if prod_key not in self.var_index:
                    return None

                phase_power = (
                    _PAULI_MUL_PHASE_POWER[axis_code][p_code]
                  + _PAULI_MUL_PHASE_POWER[axis_code][neighbor_code]
                )
                idx = self.var_index[prod_key]
                expr[idx] = expr.get(idx, 0) + 2 * h_coeff * _PAULI_PHASE[phase_power & 3]
                if expr[idx] == 0:
                    del expr[idx]

        return LinearExpr(terms=expr, const=0)

    def _add_hamil_wards(self):
        r'''
            add all representable stationarity Ward identities <[H, O]> == 0
            within the current SDP variable set

            [H,O] preserves pi-rotation charge, therefore to generate non-trivial stationarity constraints
            O must have pi-rotation charge ++, i.e. sign charge 000/111.
            Among them, only 111 sector is K-odd and gives K-even commutators.
            Given PSD variables have sign charge 000, this function should ensure that candidate O has sign charge 111.
        '''
        seen_masks = set()
        seen_keys = set()
        for pstr in self.vars:
            support = pstr.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 2
                p_code = (pstr.mask >> (2 * site)) & 3
                support &= ~(3 << (2 * site))

                for neighbor_site, axis_code, h_mask, _ in self._hamil_terms_at_site[site]:
                    if p_code == axis_code:
                        continue
                    neighbor_code = (pstr.mask >> (2 * neighbor_site)) & 3
                    if neighbor_code != 0 and neighbor_code != axis_code:
                        continue
                    mask = pstr.mask ^ h_mask
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = PauliString(self.L, mask)
                    # translation/inversion/S3-related O's give redundant Ward identities
                    # that are also related by symmetry operations.
                    # full _sym_canon would dedupe them, but it is too costly here.
                    key = self.trans_canon(cand)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_hamil_ward(cand)
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops['hamil'] += 1

    def _spin_comm(self, pstr: PauliString, axis: str) -> PauliOperator:
        '''
            return [S^a_tot, O] as a Pauli operator
        '''
        assert axis in ('X', 'Y', 'Z')
        axis_code = _PAULI_CODE[axis]
        op = PauliOperator()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            p_code = (pstr.mask >> (2*site)) & 3
            support &= ~(3 << (2*site))
            if p_code == axis_code:
                continue

            new_mask = (pstr.mask & ~(3 << (2*site))) | ((axis_code ^ p_code) << (2*site))
            coeff = _PAULI_PHASE[_PAULI_MUL_PHASE_POWER[axis_code][p_code]]
            op.add(PauliString(self.L, new_mask), coeff)
        return op

    def _compile_spin_ward(self, pstr: PauliString, axis: str) -> LinearExpr | None:
        axis_code = _PAULI_CODE[axis]
        expr = {}
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            p_code = (pstr.mask >> (2 * site)) & 3
            support &= ~(3 << (2 * site))
            if p_code == axis_code:
                continue

            mask = (pstr.mask & ~(3 << (2 * site))) | ((axis_code ^ p_code) << (2 * site))
            prod = PauliString(self.L, mask)
            if not self._sym_allowed(prod):
                continue
            prod_key = self._sym_canon(prod)
            if prod_key not in self.var_index:
                return None

            idx = self.var_index[prod_key]
            expr[idx] = expr.get(idx, 0) + _PAULI_PHASE[_PAULI_MUL_PHASE_POWER[axis_code][p_code]]
            if expr[idx] == 0:
                del expr[idx]

        return LinearExpr(terms=expr, const=0)

    def _add_spin_wards(self):
        r'''
            add all representable SO(3) Ward identities <[S^a_tot, O]> == 0
            within the current SDP variable set

            Non-trivial SO(3) Ward constraints come from commutations with charges S^a_{tot} (100, 010 and 001),
            hence ward moments must have sign charges 011, 101, or 110.
        '''
        _AXIS_BY_CHARGE = {
            (0, 1, 1): 'X',
            (1, 0, 1): 'Y',
            (1, 1, 0): 'Z',
        }

        seen_masks = set()
        seen_keys = set()
        for pstr in self.vars:
            support = pstr.mask
            while support:
                bit = support & -support
                site = (bit.bit_length() - 1) // 2
                p_code = (pstr.mask >> (2 * site)) & 3
                support &= ~(3 << (2 * site))

                for axis_code in _PAULI_CODE.values():
                    o_code = axis_code ^ p_code
                    if o_code == 0 or o_code == axis_code:
                        continue
                    mask = (pstr.mask & ~(3 << (2 * site))) | (o_code << (2 * site))
                    if mask in seen_masks:
                        continue
                    seen_masks.add(mask)

                    cand = PauliString(self.L, mask)
                    ward_axis = _AXIS_BY_CHARGE.get(cand.sign_charge())
                    if ward_axis is None:
                        continue
                    # translation/inversion/S3-related O's give redundant Ward identities
                    # that are also related by symmetry operations.
                    # full _sym_canon would dedupe them, but it is too costly here.
                    key = self.trans_canon(cand)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    expr = self._compile_spin_ward(cand, ward_axis)
                    if expr is None or not expr.terms:
                        continue
                    self.affines.add(expr)
                    self.ward_ops[f'S{ward_axis.lower()}'] += 1

    def _build_affines(self):
        self.affines = AffineConstraints(n_vars=len(self.vars))
        for key in self.ward_ops:
            self.ward_ops[key] = 0

        id_key = self._sym_canon(PauliString.from_str('I'*self.L))
        if id_key not in self.var_index:
            raise ValueError('current basis cannot represent the identity operator')
        self.affines.add(LinearExpr(terms={self.var_index[id_key]: 1}, const=-1))

        # stationarity constraints
        self._add_hamil_wards()

        # SO(3) Ward identities
        self._add_spin_wards()

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
        if self.hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        self._build_affines()

    def descendants(self, pstr: PauliString):
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

    params = HeisenbergParams(L=8, J1=1., J2=1.)
    compiler = HeisenbergCompiler(params)
    compiler.compile(build_basis_reprs(params.L, ['I', 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XXX', 'YYY', 'ZZZ']))

    print(*compiler.basis_reprs)
    print(compiler.summary())

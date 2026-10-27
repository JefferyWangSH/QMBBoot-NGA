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


def build_zz(params: IsingParams, r: int):
    assert 0 <= r <= params.L//2
    pstr = PauliString.from_str('I'*params.L) if r == 0 else PauliString.from_str('Z' + 'I'*(r-1) + 'Z' + 'I'*(params.L-r-1))
    return PauliOperator({pstr: 1.})


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
        block_keys:    symmetry sector identifying each PSD block across compilations
        psd_blocks:    momentum PSD blocks
        affines:       affine constraints
        affines_mat:   affine constraints in terms of a sparse matrix
    '''
    var_cpx: bool = False
    var_index: dict[int, int]
    vars: list[PauliString]
    _moments_lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ward_ops: dict[str, int]

    block_reprs: list[list[PauliString]]
    block_momenta: list[int]
    block_keys: list[int]
    psd_blocks: list[PSDConstraints]

    affines: AffineConstraints
    affines_mat: sp.sparse.csr_matrix

    hamil_op: PauliOperator
    _hamil_lookup: list[list[tuple[tuple[tuple[int, int], ...], int, float | complex]]]

    # SDP objective
    obj_op: PauliOperator
    obj_expr: LinearExpr
    obj_sense: str
    e_lb: float | None
    e_ub: float | None
    energy_ineqs: list[LinearExpr]

    # uncertified observables
    obs_ops: dict[str, PauliOperator | list[PauliOperator]]
    obs_exprs: dict[str, LinearExpr | list[LinearExpr]]

    def __init__(self,
        params: IsingParams,
        *,
        obj_op: PauliOperator | None = None,
        obj_sense: str = 'min',
        e_lb: float = None,
        e_ub: float = None,
        obs_ops: dict | None = None,
    ):
        self.L = params.L
        self.params = params
        self.ward_ops = {'hamil': 0}

        self.hamil_op = build_hamil(params)

        # _hamil_lookup[site] lists the local Hamiltonian terms acting on the site;
        # each entry stores
        # 1) h_sites: tuple of (site, Pauli code)
        # 2) h_mask: Hamiltonian Pauli mask
        # 3) h_coeff: Hamiltonian coefficient
        self._hamil_lookup = [[] for _ in range(self.L)]
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
                self._hamil_lookup[site].append((h_sites, h_mask, self.L * h_coeff))

        if obj_sense not in ('min', 'max'):
            raise ValueError('objective sense must be min or max')
        self.obj_sense = obj_sense

        self.obj_op = self.hamil_op if obj_op is None else obj_op
        if self.obj_op != self.hamil_op:
            if e_lb is None or e_ub is None:
                raise ValueError('observable objective requires energy bounds e_lb and e_ub')
            self.e_lb = e_lb
            self.e_ub = e_ub
        else:
            if self.obj_sense != 'min':
                raise ValueError("Hamiltonian objective requires obj_sense='min'")
            if e_lb is not None or e_ub is not None:
                raise ValueError('e_lb and e_ub are only valid for non-Hamiltonian objectives')
            self.e_lb = None
            self.e_ub = None

        self.obs_ops = {} if obs_ops is None else obs_ops.copy()

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

        orbits = [pstr, pstr.invert()]
        key = min(image.trans_canon for image in orbits)
        for image in orbits:
            self._sym_canon_cache[image.mask] = key
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
        r'''
            build K-even SDP variables

            SDP variables involve only K-even Pauli strings because:

                1) the expectation value of K-odd Pauli string, which involves odd number of Y,
                   is purely imaginary given a K-symmetric denisty matrix.

                2) any Pauli string is hermitian so its expectation should be real.

            combining these facts yields <O_odd> = 0,
            therefore any K-odd Pauli string can be removed from SDP variables.

            The variables are built in one pass over (O_b, O_a, r); the lookup (b, a, r) -> (var_idx, coeff)
            with (T_r O_a)^\dag O_b = coeff * m[var_idx] is kept in self._moments_lookup for _build_psd,
            which only adds the Fourier phases.
        '''
        self.vars = []
        self.var_index = {}
        translates = [[pstr.translate(r) for r in range(self.L)] for pstr in self.basis_reprs]

        r'''
            for each pair (O_a, O_b) we record a list of terms (r, var_idx, coeff), one per translation r:

                (T_r O_a)^\dag O_b = coeff * m[var_idx],   coeff = +-1 or +-i;

            an r whose product is not _sym_allowed gives no term.
            Only the pairs a >= b (lower triangle) are recorded because M(q) is Hermitian.

            This is a 3d array [b, a, r] with the missing r removed, flattened to 1d (as in CSR):
            the terms of all pairs are concatenated in the order (b, a, r) into three 1d arrays of the same length,

                rs[t], idxs[t], coeffs[t]   (translation, variable index, coefficient),

            and the terms of the pair (a, b) are t in [pair_offsets[k], pair_offsets[k] + pair_sizes[k]),
            k = b*n + a, n = len(basis_reprs).

            While building:
                b_rs, b_idxs, b_coeffs   terms of the current O_b (all O_a), Python lists
                rs, idxs, coeffs         one numpy chunk per O_b (converted after each O_b to save memory),
                                         concatenated at the end
                pair_offsets, pair_sizes index of the first term and number of terms of each pair, order b*n + a
                n_terms                  number of terms before the current O_b
        '''
        pair_offsets, pair_sizes, rs, idxs, coeffs = [], [], [], [], []
        n_terms = 0
        for b, pstr_b in enumerate(self.basis_reprs):
            b_rs, b_idxs, b_coeffs = [], [], []
            for a, a_translates in enumerate(translates):
                start = len(b_rs)
                pair_offsets.append(n_terms + start)
                if a < b:
                    # lower triangle only
                    pair_sizes.append(0)
                    continue
                for r, pstr_ar in enumerate(a_translates):
                    pstr, coeff = pstr_ar.dag().mul(pstr_b)

                    if not self._sym_allowed(pstr):
                        continue
                    key = self._sym_canon(pstr)
                    if key not in self.var_index:
                        self.var_index[key] = len(self.vars)
                        self.vars.append(PauliString(L=self.L, mask=key))
                    b_rs.append(r)
                    b_idxs.append(self.var_index[key])
                    b_coeffs.append(coeff)
                pair_sizes.append(len(b_rs) - start)
            rs.append(np.array(b_rs, dtype=np.int32))
            idxs.append(np.array(b_idxs, dtype=np.int32))
            coeffs.append(np.array(b_coeffs, dtype=complex))
            n_terms += len(b_rs)

        self._moments_lookup = (
            np.array(pair_offsets, dtype=np.int64),
            np.array(pair_sizes, dtype=np.int64),
            np.concatenate(rs),
            np.concatenate(idxs),
            np.concatenate(coeffs),
        )

    def fourier_phase(self, q: int, r: int) -> complex:
        # q is the integer momentum-sector index;
        # exact at multiples of L/4 and n -> -n symmetric, so Fourier sums cancel exactly
        n = (q * r) % self.L
        if 4 * n % self.L == 0:
            return 1j ** (4 * n // self.L)
        if 2 * n > self.L:
            n -= self.L
        return np.exp(1j * 2 * np.pi * n / self.L)

    @staticmethod
    def nonzero_fourier(pstr: PauliString, q: int) -> bool:
        return (q * pstr.period) % pstr.L == 0

    def _build_block_reprs(self):
        '''
            for a pauli string with period L_a < L,
            allowed momentum satisfy e^{-i k L_a} = 1 such that n L_a/L = m
        '''
        self.block_reprs = []
        self.block_momenta = []
        self.block_keys = [] # symmetry sector q
        for q in range(self.L//2 + 1):
            self.block_reprs.append([
                pstr for pstr in self.basis_reprs
                if self.nonzero_fourier(pstr, q)
            ])
            self.block_momenta.append(q)
            self.block_keys.append(q)

    def _build_psd(self):
        r'''
            build momentum PSD blocks M(k) from self._moments_lookup of _build_vars:
            M_{ab}(q) = \sum_r e^{iqr} / L * coeff * m[var_idx] over the list of the pair (a, b),
            for the lower triangle only
        '''
        self.psd_blocks = []
        n_basis = len(self.basis_reprs)
        n_vars = len(self.vars)
        pair_offsets, pair_sizes, r_terms, var_terms, coeff_terms = self._moments_lookup
        basis_index = {pstr.mask: a for a, pstr in enumerate(self.basis_reprs)}

        r'''
            K symmetry imposes that M(-k) = diag(eta) M(k)^\ast diag(eta)
            therefore the number of independent momentum PSD blocks can be reduced by half
        '''
        for q, block_key, block_basis in zip(self.block_momenta, self.block_keys, self.block_reprs):
            dim = len(block_basis)
            psd = PSDConstraints(
                n_vars=n_vars,
                dim=dim,
                block_key=block_key,
                basis_keys=[self.trans_canon(pstr) for pstr in block_basis],
            )

            phases = np.array([self.fourier_phase(q, r) / self.L for r in range(self.L)])
            # block_basis keeps the order of basis_reprs, so the lower triangle (row >= col) has a >= b
            block_index = np.array([basis_index[pstr.mask] for pstr in block_basis])
            if np.any(np.diff(block_index) <= 0):
                raise ValueError('block_basis must follow the order of basis_reprs without duplicates')
            rows, cols = np.tril_indices(dim)
            pairs = block_index[cols] * n_basis + block_index[rows]
            starts = pair_offsets[pairs]
            counts = pair_sizes[pairs]
            terms = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)

            # COO terms of the affine map vec(M(q)) = A @ m, lower triangle
            psd.rows = np.repeat(rows + cols*dim, counts)
            psd.cols = var_terms[terms]
            psd.vals = phases[r_terms[terms]] * coeff_terms[terms]

            # at complex-conjugation invariant momenta (q = -q mod L),
            # K maps M(k) to itself and D^H M(k) D is real for D = diag(i^{n_a}),
            # n_a = parity of the number of Y in P_a
            if (2 * q) % self.L == 0:
                psd.realify_phase = [pstr.parity() for pstr in block_basis]
            self.psd_blocks.append(psd)

    def _hamil_comm(self, pstr: PauliString) -> PauliOperator:
        op = PauliOperator()
        seen = set()
        support = pstr.mask
        while support:
            bit = support & -support
            site = (bit.bit_length() - 1) // 2
            support &= ~(3 << (2*site))

            for h_sites, h_mask, h_coeff in self._hamil_lookup[site]:
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

            for h_sites, h_mask, h_coeff in self._hamil_lookup[site]:
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

                for h_sites, h_mask, _ in self._hamil_lookup[site]:
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

        self.obj_expr = self._compile_expr(self.obj_op)
        if self.obj_expr is None:
            raise ValueError('current basis cannot represent the SDP objective')

        self.energy_ineqs = []
        hamil_expr = self._compile_expr(self.hamil_op)
        if hamil_expr is None:
            raise ValueError('current basis cannot represent the Hamiltonian')

        if self.e_lb is not None:
            self.energy_ineqs.append(LinearExpr(
                terms=hamil_expr.terms.copy(),
                const=hamil_expr.const - self.e_lb,
            ))
        if self.e_ub is not None:
            self.energy_ineqs.append(LinearExpr(
                terms={idx: -coeff for idx, coeff in hamil_expr.terms.items()},
                const=self.e_ub - hamil_expr.const,
            ))

        self.obs_exprs = {}
        for name, obs in self.obs_ops.items():
            if isinstance(obs, list):
                exprs = []
                for idx, op in enumerate(obs):
                    expr = self._compile_expr(op)
                    if expr is None:
                        raise ValueError(f'current basis cannot represent observable: {name}[{idx}] = {op}')
                    exprs.append(expr)
                self.obs_exprs[name] = exprs
            else:
                expr = self._compile_expr(obs)
                if expr is None:
                    raise ValueError(f'current basis cannot represent observable: {name} = {obs}')
                self.obs_exprs[name] = expr

        self._build_block_reprs()
        self._build_psd()
        self._moments_lookup = None
        self._build_affines()

    def descendants(self, pstr: PauliString):
        r'''
            calculate

                [H, O_a(0)] = \sum_{\mu,s} C_{a\mu}(s) T_s O_\mu(0)

            as a list [(O_\mu(0), s, coeff), ...]
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
            'obj_expr': self._get_expr_str(self.obj_expr),
            'obj_sense': self.obj_sense,
            'obs_exprs': {
                name: [self._get_expr_str(item) for item in expr] if isinstance(expr, list) else self._get_expr_str(expr)
                for name, expr in self.obs_exprs.items()
            },
            'energy_ineqs': [self._get_expr_str(ineq) for ineq in self.energy_ineqs],
        }

    def sdp_data(self):
        return SDPData(
            var_cpx = self.var_cpx,
            n_vars = len(self.vars),
            objective = self.obj_expr,
            objective_sense = self.obj_sense,
            psd_blocks = self.psd_blocks,
            affines_mat = self.affines_mat,
            observables = self.obs_exprs,
            energy_ineqs = self.energy_ineqs,
            var_keys = sorted(self.var_index, key=self.var_index.get),
        )

    def clone(self):
        return type(self)(
            self.params,
            obj_op=self.obj_op,
            obj_sense=self.obj_sense,
            e_lb=self.e_lb,
            e_ub=self.e_ub,
            obs_ops=self.obs_ops,
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
    print(compiler.summary())

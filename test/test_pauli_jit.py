from operators.pauli import PauliString as PyPauliString
from operators.pauli import PauliOperator
from operators.pauli_jit import PauliString as CppPauliString

STRING_CASES = [
    'I',
    'X',
    'Y',
    'Z',
    'IIII',
    'IXYZ',
    'XYZI',
    'ZZXX',
    'IYZX',
    'IXYZIX',
    'XYZXYZ',
]

MASK_CASES = {
    1: [0, 1, 2, 3],
    4: [0, 1, 2, 3, 0b01101100, 0b11100100],
    16: [0, 1, (1 << 31), (1 << 0) | (1 << 10) | (1 << 31)],
    36: [0, 1, (1 << 63) | (1 << 64), (1 << 0) | (1 << 41) | (1 << 71)],
}

PERMS = [
    ('X', 'Y', 'Z'),
    ('X', 'Z', 'Y'),
    ('Y', 'X', 'Z'),
    ('Y', 'Z', 'X'),
    ('Z', 'X', 'Y'),
    ('Z', 'Y', 'X'),
]

def assert_same_pstr(cpp_p, py_p):
    assert cpp_p.L == py_p.L
    assert cpp_p.mask == py_p.mask
    assert str(cpp_p) == str(py_p)
    assert repr(cpp_p) == f'PauliString(L={py_p.L}, mask={py_p.mask})'
    assert hash(cpp_p) == hash(py_p)
    assert cpp_p == CppPauliString(py_p.L, py_p.mask)
    assert py_p == PyPauliString(cpp_p.L, cpp_p.mask)
    assert cpp_p.trans_canon == py_p.trans_canon
    assert cpp_p.trans_canon_rep.L == py_p.trans_canon_rep.L
    assert cpp_p.trans_canon_rep.mask == py_p.trans_canon_rep.mask
    assert cpp_p.period == py_p.period
    assert cpp_p.parity() == py_p.parity()
    assert cpp_p.sign_charge() == py_p.sign_charge()
    assert cpp_p.pi_rot_charge() == py_p.pi_rot_charge()

def assert_same_operator(cpp_op, py_op):
    assert cpp_op.L == py_op.L
    cpp_terms = {(p.L, p.mask): c for p, c in cpp_op.terms.items()}
    py_terms = {(p.L, p.mask): c for p, c in py_op.terms.items()}
    assert cpp_terms == py_terms
    assert str(cpp_op) == str(py_op)

def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f'{exc_type.__name__} was not raised')


if __name__ == '__main__':

    # construction and formatting
    for L, masks in MASK_CASES.items():
        assert_same_pstr(CppPauliString.identity(L), PyPauliString(L))
        assert_same_pstr(CppPauliString(L), PyPauliString(L))

        for mask in masks:
            assert_same_pstr(CppPauliString(L, mask), PyPauliString(L, mask))

    for s in STRING_CASES:
        assert_same_pstr(CppPauliString.from_str(s), PyPauliString.from_str(s))

    # lattice and internal symmetries
    shifts = [-37, -36, -35, -5, -1, 0, 1, 5, 35, 36, 37]
    for L, masks in MASK_CASES.items():
        for mask in masks:
            cpp_p = CppPauliString(L, mask)
            py_p = PyPauliString(L, mask)

            for shift in shifts:
                assert_same_pstr(cpp_p.translate(shift), py_p.translate(shift))

            assert_same_pstr(cpp_p.invert(), py_p.invert())
            for perm in PERMS:
                assert_same_pstr(cpp_p.permute(perm), py_p.permute(perm))

    # multiplication and dag
    for L, masks in MASK_CASES.items():
        for left in masks:
            for right in masks:
                cpp_prod, cpp_phase = CppPauliString(L, left).mul(CppPauliString(L, right))
                py_prod, py_phase = PyPauliString(L, left).mul(PyPauliString(L, right))
                assert_same_pstr(cpp_prod, py_prod)
                assert cpp_phase == py_phase
                assert_same_pstr(CppPauliString(L, left).dag(), PyPauliString(L, left).dag())

    # Python PauliOperator with Python/C++ string keys
    L = 4

    cpp_a = PauliOperator({CppPauliString.from_str('IXYZ'): 2.0})
    cpp_a.add(CppPauliString.from_str('ZZXX'), -1.0)
    cpp_b = PauliOperator({CppPauliString.from_str('XYZI'): 3.0 + 2.0j})
    cpp_b.add(CppPauliString.from_str('IYZX'), -4.0j)

    py_a = PauliOperator({PyPauliString.from_str('IXYZ'): 2.0})
    py_a.add(PyPauliString.from_str('ZZXX'), -1.0)
    py_b = PauliOperator({PyPauliString.from_str('XYZI'): 3.0 + 2.0j})
    py_b.add(PyPauliString.from_str('IYZX'), -4.0j)

    assert_same_operator(cpp_a, py_a)
    assert_same_operator(cpp_b, py_b)
    assert_same_operator(cpp_a + cpp_b, py_a + py_b)
    assert_same_operator(cpp_a - cpp_b, py_a - py_b)
    assert_same_operator(-cpp_a, -py_a)
    assert_same_operator((2.0 - 1.0j) * cpp_b, (2.0 - 1.0j) * py_b)
    assert_same_operator(cpp_a.mul(cpp_b), py_a.mul(py_b))
    assert_same_operator(cpp_b.dag(), py_b.dag())
    assert_same_operator(cpp_a.commutator(cpp_b), py_a.commutator(py_b))

    cpp_zero = PauliOperator({CppPauliString.from_str('IXYZ'): 1.0})
    cpp_zero.add(CppPauliString.from_str('IXYZ'), -1.0)
    py_zero = PauliOperator({PyPauliString.from_str('IXYZ'): 1.0})
    py_zero.add(PyPauliString.from_str('IXYZ'), -1.0)
    assert_same_operator(cpp_zero, py_zero)
    assert cpp_zero == PauliOperator()
    assert py_zero == PauliOperator()

    cpp_empty = PauliOperator()
    py_empty = PauliOperator()
    assert_same_operator(cpp_empty + cpp_a, py_empty + py_a)
    assert_same_operator(cpp_a + cpp_empty, py_a + py_empty)
    assert_same_operator(cpp_empty - cpp_a, py_empty - py_a)
    assert_same_operator(cpp_a - cpp_empty, py_a - py_empty)
    assert_same_operator(cpp_empty.mul(cpp_a), py_empty.mul(py_a))
    assert_same_operator(cpp_a.mul(cpp_empty), py_a.mul(py_empty))
    assert_same_operator(cpp_empty.dag(), py_empty.dag())
    assert_same_operator(cpp_empty.commutator(cpp_a), py_empty.commutator(py_a))

    # error paths
    assert_raises(ValueError, lambda: CppPauliString(0))
    assert_raises(ValueError, lambda: CppPauliString(4, -1))
    assert_raises(ValueError, lambda: CppPauliString(4, 1 << 8))
    assert_raises(ValueError, lambda: CppPauliString.from_str('IXA'))
    assert_raises(ValueError, lambda: CppPauliString(4, 1).permute(('X', 'Y')))
    assert_raises(ValueError, lambda: CppPauliString(4, 1).permute(('X', 'Y', 'A')))
    assert_raises(TypeError, lambda: CppPauliString(4, 1).mul(CppPauliString(5, 1)))

    print('PauliString JIT equivalence test passed')

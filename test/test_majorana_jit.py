from operators.majorana import MajoranaMonomial as PyMajoranaMonomial
from operators.majorana import MajoranaOperator
from operators.majorana_jit import MajoranaMonomial as CppMajoranaMonomial

MONOMIAL_CASES = {
    1: [
        0,
        1 << 0,
        1 << 1,
        (1 << 0) | (1 << 3),
    ],
    6: [
        0,
        1 << 0,
        (1 << 1) | (1 << 2),
        (1 << 4) | (1 << 5) | (1 << 20) | (1 << 23),
    ],
    16: [
        0,
        (1 << 0) | (1 << 63),
        (1 << 1) | (1 << 2) | (1 << 60) | (1 << 61),
    ],
    36: [
        0,
        1 << 0,
        (1 << 63) | (1 << 64),
        (1 << 0) | (1 << 65) | (1 << 143),
        (1 << 1) | (1 << 2) | (1 << 67) | (1 << 130) | (1 << 143),
        (1 << 4) | (1 << 5) | (1 << 68) | (1 << 69) | (1 << 132) | (1 << 133),
    ],
}

def assert_same_monomial(cpp_m, py_m):
    assert cpp_m.L == py_m.L
    assert cpp_m.mask == py_m.mask
    assert str(cpp_m) == str(py_m)
    assert hash(cpp_m) == hash(py_m)
    assert cpp_m == CppMajoranaMonomial(py_m.L, py_m.mask)
    assert py_m == PyMajoranaMonomial(cpp_m.L, cpp_m.mask)
    assert cpp_m.degree() == py_m.degree()
    assert cpp_m.dag_phase() == py_m.dag_phase()
    assert cpp_m.hermitian_phase() == py_m.hermitian_phase()
    assert cpp_m.fermion_parity() == py_m.fermion_parity()
    assert cpp_m.fermion_parity(spin=True) == py_m.fermion_parity(spin=True)
    assert cpp_m.k_parity() == py_m.k_parity()
    assert cpp_m.k_parity(hermitian=True) == py_m.k_parity(hermitian=True)
    assert cpp_m.k_parity(hermitian=False) == py_m.k_parity(hermitian=False)

def assert_same_operator(cpp_op, py_op):
    assert cpp_op.L == py_op.L
    cpp_terms = {(m.L, m.mask): c for m, c in cpp_op.terms.items()}
    py_terms = {(m.L, m.mask): c for m, c in py_op.terms.items()}
    assert cpp_terms == py_terms
    assert str(cpp_op) == str(py_op)

def assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f'{exc_type.__name__} was not raised')


if __name__ == '__main__':

    # construction and basic properties
    for L, masks in MONOMIAL_CASES.items():
        assert_same_monomial(CppMajoranaMonomial.identity(L), PyMajoranaMonomial.identity(L))
        assert_same_monomial(CppMajoranaMonomial(L), PyMajoranaMonomial(L))

        for mask in masks:
            assert_same_monomial(CppMajoranaMonomial(L, mask), PyMajoranaMonomial(L, mask))

    # from_str and ordered-product signs
    strings = [
        '',
        'I',
        '0u+',
        '0u+ 0u+',
        '0u- 0u+',
        '0u+ 1d- 5u-',
        '5d- 0u+ 2d+',
        '16u+ 0u+ 35d-',
    ]
    for s in strings:
        cpp_result, cpp_sign = CppMajoranaMonomial.from_str(36, s, sign=True)
        py_result, py_sign = PyMajoranaMonomial.from_str(36, s, sign=True)
        assert_same_monomial(cpp_result, py_result)
        assert cpp_sign == py_sign
        assert_same_monomial(CppMajoranaMonomial.from_str(36, s), PyMajoranaMonomial.from_str(36, s))

    # lattice operations and translation-derived properties
    shifts = [-37, -36, -35, -5, -1, 0, 1, 5, 35, 36, 37]
    for L, masks in MONOMIAL_CASES.items():
        for mask in masks:
            cpp_m = CppMajoranaMonomial(L, mask)
            py_m = PyMajoranaMonomial(L, mask)

            for shift in shifts:
                cpp_t, cpp_sign = cpp_m.translate(shift)
                py_t, py_sign = py_m.translate(shift)
                assert_same_monomial(cpp_t, py_t)
                assert cpp_sign == py_sign

            assert cpp_m.trans_canon == py_m.trans_canon
            assert cpp_m.trans_canon_sign == py_m.trans_canon_sign
            assert_same_monomial(cpp_m.trans_canon_rep, py_m.trans_canon_rep)
            assert cpp_m.period == py_m.period
            assert cpp_m.period_sign == py_m.period_sign

            cpp_inv, cpp_sign = cpp_m.invert()
            py_inv, py_sign = py_m.invert()
            assert_same_monomial(cpp_inv, py_inv)
            assert cpp_sign == py_sign

    # spin/internal symmetry operations
    for L, masks in MONOMIAL_CASES.items():
        for mask in masks:
            cpp_m = CppMajoranaMonomial(L, mask)
            py_m = PyMajoranaMonomial(L, mask)

            cpp_ex, cpp_sign = cpp_m.spin_exchange()
            py_ex, py_sign = py_m.spin_exchange()
            assert_same_monomial(cpp_ex, py_ex)
            assert cpp_sign == py_sign

            for up_quarters in [-5, -1, 0, 1, 2, 3, 4, 5]:
                for dn_quarters in [-5, -1, 0, 1, 2, 3, 4, 5]:
                    cpp_rot, cpp_sign = cpp_m.c4_rotate(up_quarters, dn_quarters)
                    py_rot, py_sign = py_m.c4_rotate(up_quarters, dn_quarters)
                    assert_same_monomial(cpp_rot, py_rot)
                    assert cpp_sign == py_sign

    # monomial multiplication
    for L, masks in MONOMIAL_CASES.items():
        for left in masks:
            for right in masks:
                cpp_prod, cpp_sign = CppMajoranaMonomial(L, left).mul(CppMajoranaMonomial(L, right))
                py_prod, py_sign = PyMajoranaMonomial(L, left).mul(PyMajoranaMonomial(L, right))
                assert_same_monomial(cpp_prod, py_prod)
                assert cpp_sign == py_sign

    # Python MajoranaOperator with Python/C++ monomial keys
    L = 36
    masks = MONOMIAL_CASES[36]

    cpp_id = MajoranaOperator({CppMajoranaMonomial.identity(L): 1.0})
    py_id = MajoranaOperator({PyMajoranaMonomial.identity(L): 1.0})
    assert_same_operator(cpp_id, py_id)

    cpp_a = MajoranaOperator({CppMajoranaMonomial(L, masks[1]): 2.0})
    cpp_a.add(CppMajoranaMonomial(L, masks[2]), -1.0)
    cpp_b = MajoranaOperator({CppMajoranaMonomial(L, masks[3]): 3.0 + 2.0j})
    cpp_b.add(CppMajoranaMonomial(L, masks[4]), -4.0j)

    py_a = MajoranaOperator({PyMajoranaMonomial(L, masks[1]): 2.0})
    py_a.add(PyMajoranaMonomial(L, masks[2]), -1.0)
    py_b = MajoranaOperator({PyMajoranaMonomial(L, masks[3]): 3.0 + 2.0j})
    py_b.add(PyMajoranaMonomial(L, masks[4]), -4.0j)

    assert_same_operator(cpp_a, py_a)
    assert_same_operator(cpp_b, py_b)
    assert_same_operator(cpp_a + cpp_b, py_a + py_b)
    assert_same_operator(cpp_a - cpp_b, py_a - py_b)
    assert_same_operator(-cpp_a, -py_a)
    assert_same_operator((2.0 - 1.0j) * cpp_b, (2.0 - 1.0j) * py_b)
    assert_same_operator(cpp_a.mul(cpp_b), py_a.mul(py_b))
    assert_same_operator(cpp_b.dag(), py_b.dag())
    assert_same_operator(cpp_a.commutator(cpp_b), py_a.commutator(py_b))

    cpp_zero = MajoranaOperator({CppMajoranaMonomial(L, masks[1]): 1.0})
    cpp_zero.add(CppMajoranaMonomial(L, masks[1]), -1.0)
    py_zero = MajoranaOperator({PyMajoranaMonomial(L, masks[1]): 1.0})
    py_zero.add(PyMajoranaMonomial(L, masks[1]), -1.0)
    assert_same_operator(cpp_zero, py_zero)
    assert cpp_zero == MajoranaOperator()
    assert py_zero == MajoranaOperator()

    cpp_empty = MajoranaOperator()
    py_empty = MajoranaOperator()
    assert_same_operator(cpp_empty + cpp_a, py_empty + py_a)
    assert_same_operator(cpp_a + cpp_empty, py_a + py_empty)
    assert_same_operator(cpp_empty - cpp_a, py_empty - py_a)
    assert_same_operator(cpp_a - cpp_empty, py_a - py_empty)
    assert_same_operator(cpp_empty.mul(cpp_a), py_empty.mul(py_a))
    assert_same_operator(cpp_a.mul(cpp_empty), py_a.mul(py_empty))
    assert_same_operator(cpp_empty.dag(), py_empty.dag())
    assert_same_operator(cpp_empty.commutator(cpp_a), py_empty.commutator(py_a))
    assert_same_operator(cpp_id.mul(cpp_a), py_id.mul(py_a))
    assert_same_operator(cpp_a.mul(cpp_id), py_a.mul(py_id))
    assert_same_operator(cpp_id.dag(), py_id.dag())

    cpp_copy = cpp_a.copy()
    py_copy = py_a.copy()
    assert_same_operator(cpp_copy, py_copy)
    cpp_copy.add(CppMajoranaMonomial(L, masks[1]), -2.0)
    py_copy.add(PyMajoranaMonomial(L, masks[1]), -2.0)
    assert_same_operator(cpp_copy, py_copy)

    cpp_scalar_zero = 0 * cpp_a
    py_scalar_zero = 0 * py_a
    assert_same_operator(cpp_scalar_zero, py_scalar_zero)

    # error paths
    assert_raises(ValueError, lambda: CppMajoranaMonomial(0))
    assert_raises(ValueError, lambda: CppMajoranaMonomial(4, -1))
    assert_raises(ValueError, lambda: CppMajoranaMonomial(4, 1 << 16))
    assert_raises(ValueError, lambda: CppMajoranaMonomial.from_str(4, 'u+'))
    assert_raises(ValueError, lambda: CppMajoranaMonomial.from_str(4, '0x+'))
    assert_raises(ValueError, lambda: CppMajoranaMonomial.from_str(4, '0u?'))
    assert_raises(TypeError, lambda: CppMajoranaMonomial(4, 1).mul(CppMajoranaMonomial(5, 1)))

    print('Majorana USE_JIT=0/1 equivalence test passed')

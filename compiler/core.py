from functools import lru_cache
from pathlib import Path

from operators.jit import load_module, MajoranaMonomial, MajoranaMonomialSquare, PauliString

__all__ = ['HubbardCompilerCore', 'HubbardSquareCompilerCore', 'HeisenbergCompilerCore']

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CPP_PATH = _REPO_ROOT / 'compiler' / 'cpp'
_OPERATORS_CPP_PATH = _REPO_ROOT / 'operators' / 'cpp'
_CACHE_PATH = _REPO_ROOT / '.cache' / 'compiler' / 'core'


@lru_cache(maxsize=None)
def _hubbard(L: int):
    if L <= 0:
        raise ValueError('L must be positive')

    # register the MajoranaMonomial<L> pybind type first
    MajoranaMonomial.identity(L)

    return load_module(
        'hubbard_core',
        _CPP_PATH / 'bindings' / 'hubbard.cpp.in',
        (
            _CPP_PATH / 'core' / 'common.hpp',
            _CPP_PATH / 'core' / 'hubbard.hpp',
            _OPERATORS_CPP_PATH / 'majorana.hpp',
            _OPERATORS_CPP_PATH / 'words.hpp',
        ),
        _CACHE_PATH / 'hubbard',
        L=L,
    )


@lru_cache(maxsize=None)
def _hubbard_square(Lx: int, Ly: int):
    if Lx <= 0 or Ly <= 0:
        raise ValueError('Lx and Ly must be positive')

    # register the MajoranaMonomialSquare<Lx, Ly> pybind type first
    MajoranaMonomialSquare.identity(Lx, Ly)

    return load_module(
        'hubbard_square_core',
        _CPP_PATH / 'bindings' / 'hubbard_square.cpp.in',
        (
            _CPP_PATH / 'core' / 'common.hpp',
            _CPP_PATH / 'core' / 'hubbard_square.hpp',
            _OPERATORS_CPP_PATH / 'majorana.hpp',
            _OPERATORS_CPP_PATH / 'majorana_square.hpp',
            _OPERATORS_CPP_PATH / 'words.hpp',
        ),
        _CACHE_PATH / 'hubbard_square',
        Lx=Lx,
        Ly=Ly,
    )


@lru_cache(maxsize=None)
def _heisenberg(L: int):
    if L <= 0:
        raise ValueError('L must be positive')

    # register the PauliString<L> pybind type first
    PauliString.identity(L)

    return load_module(
        'heisenberg_core',
        _CPP_PATH / 'bindings' / 'heisenberg.cpp.in',
        (
            _CPP_PATH / 'core' / 'common.hpp',
            _CPP_PATH / 'core' / 'heisenberg.hpp',
            _OPERATORS_CPP_PATH / 'pauli.hpp',
            _OPERATORS_CPP_PATH / 'words.hpp',
        ),
        _CACHE_PATH / 'heisenberg',
        L=L,
    )


class HubbardCompilerCore:
    def __new__(cls, L: int, hamil_monomials: list, hamil_coeffs: list, use_ph_sym: bool):
        return _hubbard(L).HubbardCompilerCore(hamil_monomials, hamil_coeffs, use_ph_sym)


class HubbardSquareCompilerCore:
    def __new__(cls, Lx: int, Ly: int, hamil_monomials: list, hamil_coeffs: list, use_ph_sym: bool):
        return _hubbard_square(Lx, Ly).HubbardSquareCompilerCore(hamil_monomials, hamil_coeffs, use_ph_sym)


class HeisenbergCompilerCore:
    def __new__(cls, L: int, hamil_paulis: list, hamil_coeffs: list):
        return _heisenberg(L).HeisenbergCompilerCore(hamil_paulis, hamil_coeffs)

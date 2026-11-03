from functools import lru_cache
from pathlib import Path

from operators.jit import load_module, MajoranaMonomial

__all__ = ['HubbardCompilerCore']

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


class HubbardCompilerCore:
    def __new__(cls, L: int, hamil_monomials: list, hamil_coeffs: list, use_ph_sym: bool):
        return _hubbard(L).HubbardCompilerCore(hamil_monomials, hamil_coeffs, use_ph_sym)

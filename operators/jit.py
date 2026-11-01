from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import cppimport

__all__ = ['load_module', 'MajoranaMonomial', 'MajoranaMonomialSquare', 'PauliString']

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CPP_PATH = _REPO_ROOT / 'operators' / 'cpp'
_CACHE_PATH = _REPO_ROOT / '.cache' / 'operators'


@lru_cache(maxsize=None)
def load_module(prefix: str, template: Path, headers: tuple[Path, ...], cache_dir: Path, **sizes: int):
    # The hash covers the template and every header it includes,
    # so editing any of them builds a new module.
    h = hashlib.sha256()
    for path in (template, *headers):
        h.update(path.read_bytes())
    tag = '_'.join(f'{key}{size}' for key, size in sizes.items())
    name = f'_{prefix}_{tag}_{h.hexdigest()[:12]}'

    # Each size fills the placeholder of its upper-case key, e.g. Lx fills @LX@.
    source = template.read_text().replace('@REPO_ROOT@', str(_REPO_ROOT)).replace('@MODULE_NAME@', name)
    for key, size in sizes.items():
        source = source.replace(f'@{key.upper()}@', str(size))

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{name}.cpp'
    if not path.exists() or path.read_text() != source:
        path.write_text(source)
    importlib.invalidate_caches()
    return cppimport.imp_from_filepath(str(path), fullname=name)


@lru_cache(maxsize=None)
def _majorana(L: int):
    if L <= 0:
        raise ValueError('L must be positive')
    return load_module(
        'majorana',
        _CPP_PATH / 'bindings' / 'majorana.cpp.in',
        (_CPP_PATH / 'majorana.hpp', _CPP_PATH / 'py_mask.hpp'),
        _CACHE_PATH / 'majorana',
        L=L,
    )


@lru_cache(maxsize=None)
def _majorana_square(Lx: int, Ly: int):
    if Lx <= 0 or Ly <= 0:
        raise ValueError('Lx and Ly must be positive')
    return load_module(
        'majorana_square',
        _CPP_PATH / 'bindings' / 'majorana_square.cpp.in',
        (_CPP_PATH / 'majorana_square.hpp', _CPP_PATH / 'py_mask.hpp'),
        _CACHE_PATH / 'majorana_square',
        Lx=Lx,
        Ly=Ly,
    )


@lru_cache(maxsize=None)
def _pauli(L: int):
    if L <= 0:
        raise ValueError('L must be positive')
    return load_module(
        'pauli',
        _CPP_PATH / 'bindings' / 'pauli.cpp.in',
        (_CPP_PATH / 'pauli.hpp', _CPP_PATH / 'py_mask.hpp'),
        _CACHE_PATH / 'pauli',
        L=L,
    )


class MajoranaMonomial:
    def __new__(cls, L: int, mask: int = 0):
        return _majorana(L).MajoranaMonomial(L, mask)

    @staticmethod
    def from_str(L: int, s: str, sign: bool = False):
        return _majorana(L).MajoranaMonomial.from_str(L, s, sign=sign)

    @staticmethod
    def identity(L: int):
        return _majorana(L).MajoranaMonomial.identity(L)


class MajoranaMonomialSquare:
    def __new__(cls, Lx: int, Ly: int, mask: int = 0):
        return _majorana_square(Lx, Ly).MajoranaMonomialSquare(Lx, Ly, mask)

    @staticmethod
    def from_str(Lx: int, Ly: int, s: str, sign: bool = False):
        return _majorana_square(Lx, Ly).MajoranaMonomialSquare.from_str(Lx, Ly, s, sign=sign)

    @staticmethod
    def identity(Lx: int, Ly: int):
        return _majorana_square(Lx, Ly).MajoranaMonomialSquare.identity(Lx, Ly)


class PauliString:
    def __new__(cls, L: int, mask: int = 0):
        return _pauli(L).PauliString(L, mask)

    @staticmethod
    def from_str(pstr: str):
        return _pauli(len(pstr)).PauliString.from_str(pstr)

    @staticmethod
    def identity(L: int):
        return _pauli(L).PauliString.identity(L)

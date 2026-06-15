from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import cppimport

from operators.majorana_square_jit import MajoranaMonomialSquare

__all__ = ['sym_canon']

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_PATH = _REPO_ROOT / 'compiler' / 'cpp' / 'hubbard_square_kernel_binding.cpp.in'
_CACHE_ROOT = _REPO_ROOT / '.cache' / 'compiler' / 'kernels' / 'hubbard_square'


def _source_hash() -> str:
    h = hashlib.sha256()
    for path in (
        _TEMPLATE_PATH,
        _REPO_ROOT / 'compiler' / 'cpp' / 'hubbard_square_kernel.hpp',
        _REPO_ROOT / 'operators' / 'cpp' / 'majorana_square.hpp',
        _REPO_ROOT / 'operators' / 'cpp' / 'py_mask.hpp',
    ):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _module_name(Lx: int, Ly: int) -> str:
    return f'_hubbard_square_kernel_Lx{Lx}_Ly{Ly}_{_source_hash()}'


def _source_path(module_name: str) -> Path:
    return _CACHE_ROOT / f'{module_name}.cpp'


def _render_source(Lx: int, Ly: int, module_name: str) -> str:
    return (
        _TEMPLATE_PATH.read_text()
        .replace('@REPO_ROOT@', str(_REPO_ROOT))
        .replace('@MODULE_NAME@', module_name)
        .replace('@LX@', str(Lx))
        .replace('@LY@', str(Ly))
    )


@lru_cache(maxsize=None)
def _load_module(Lx: int, Ly: int):
    if Lx <= 0:
        raise ValueError('Lx must be positive')
    if Ly <= 0:
        raise ValueError('Ly must be positive')

    # register the MajoranaMonomialSquare<Lx, Ly> pybind type first
    MajoranaMonomialSquare.identity(Lx, Ly)

    module_name = _module_name(Lx, Ly)
    source_path = _source_path(module_name)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    source = _render_source(Lx, Ly, module_name)
    if not source_path.exists() or source_path.read_text() != source:
        source_path.write_text(source)

    importlib.invalidate_caches()
    return cppimport.imp_from_filepath(str(source_path), fullname=module_name)


def sym_canon(monomial, sign: bool = False):
    mod = _load_module(monomial.Lx, monomial.Ly)
    return mod.sym_canon(monomial, sign=sign)

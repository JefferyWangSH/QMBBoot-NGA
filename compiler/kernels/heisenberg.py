from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import cppimport

from operators.pauli_jit import PauliString

__all__ = ['sym_canon']

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_PATH = _REPO_ROOT / 'compiler' / 'cpp' / 'heisenberg_kernel_binding.cpp.in'
_CACHE_ROOT = _REPO_ROOT / '.cache' / 'compiler' / 'kernels' / 'heisenberg'


def _source_hash() -> str:
    h = hashlib.sha256()
    for path in (
        _TEMPLATE_PATH,
        _REPO_ROOT / 'compiler' / 'cpp' / 'heisenberg_kernel.hpp',
        _REPO_ROOT / 'operators' / 'cpp' / 'pauli.hpp',
        _REPO_ROOT / 'operators' / 'cpp' / 'py_mask.hpp',
    ):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _module_name(L: int) -> str:
    return f'_heisenberg_kernel_L{L}_{_source_hash()}'


def _source_path(module_name: str) -> Path:
    return _CACHE_ROOT / f'{module_name}.cpp'


def _render_source(L: int, module_name: str) -> str:
    return (
        _TEMPLATE_PATH.read_text()
        .replace('@REPO_ROOT@', str(_REPO_ROOT))
        .replace('@MODULE_NAME@', module_name)
        .replace('@L@', str(L))
    )


@lru_cache(maxsize=None)
def _load_module(L: int):
    if L <= 0:
        raise ValueError('L must be positive')

    # register the PauliString<L> pybind type first
    PauliString.identity(L)

    module_name = _module_name(L)
    source_path = _source_path(module_name)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    source = _render_source(L, module_name)
    if not source_path.exists() or source_path.read_text() != source:
        source_path.write_text(source)

    importlib.invalidate_caches()
    return cppimport.imp_from_filepath(str(source_path), fullname=module_name)


def sym_canon(pstr):
    mod = _load_module(pstr.L)
    return mod.sym_canon(pstr)

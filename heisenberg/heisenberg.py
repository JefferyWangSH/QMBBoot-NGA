from dataclasses import dataclass

from ising.ising import (
    PauliString, IsingOperator, IsingCompiler,
    build_basis_reprs as _build_basis_reprs,
)

build_basis_reprs = _build_basis_reprs


@dataclass(slots=True)
class HeisenbergParams:
    L: int = 8
    J1: float = 1.
    J2: float = 1.


def _two_site_op(L: int, pauli: str, dist: int):
    return PauliString.from_str(pauli + 'I'*(dist-1) + pauli + 'I'*(L-dist-1))


def build_hamil(params: HeisenbergParams):
    assert params.L >= 3
    hamil_op = IsingOperator()

    for pauli in 'XYZ':
        nn = _two_site_op(params.L, pauli, 1)
        nnn = _two_site_op(params.L, pauli, 2)
        for shift in range(params.L):
            hamil_op.add(nn.translate(shift), params.J1 / params.L)
            hamil_op.add(nnn.translate(shift), params.J2 / params.L)

    return hamil_op


class HeisenbergCompiler(IsingCompiler):
    def __init__(self, params: HeisenbergParams):
        self.L = params.L
        self.params = params
        self.hamil_op = build_hamil(params)

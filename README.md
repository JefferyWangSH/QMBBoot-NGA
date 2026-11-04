# QMBBoot-NGA
* *Nullspace-guided adaptive (NGA) algorithm for quantum many-body bootstrap ([arXiv:2609.25834](https://arxiv.org/abs/2609.25834)).*
  * Certify the ground states of periodic lattice models: lower bounds on the energy density and two-sided bounds on general static observables.
  * Iteratively improve the bounds by refining the bootstrap basis guided by the moment-matrix nullspace.
* Exploit symmetries to
  * reveal block-diagonal structure in PSD moment matrices,
  * prune SDP optimization variables,
  * and impose Ward-identity constraints.
* Run symbolic operator algebra and SDP compilation in C++ (JIT-compiled with cppimport and OpenMP).
* Use [scs-qmbboot](https://github.com/JefferyWangSH/scs-qmbboot), a fork of [SCS](https://github.com/cvxgrp/scs) specialized for quantum many-body bootstrap, by default to solve SDPs; also support various SDP backends through [CVXPY](https://github.com/cvxpy/cvxpy).
* Enable *warm start* between NGA steps when using [scs-qmbboot](https://github.com/JefferyWangSH/scs-qmbboot), which dramatically accelerates the NGA trajectory.

## Models
The following models are implemented:
* *Transverse- and longitudinal-field Ising chain*
* *$J_1$-$J_2$ Heisenberg chain*
* *Hubbard chain*
* *Hubbard models on square/rectangular lattices*

All these model compilers use:
* lattice translation symmetry to generate momentum PSD blocks
* lattice point-group symmetries to prune SDP variables, e.g. inversion in 1D and $D_4$/$D_2$ for square/rectangular lattices
* complex conjugation $\mathcal{K}$-symmetry to prune $\mathcal{K}$-odd SDP variables and identify equivalent $k$ and $-k$ momentum PSD blocks

Additional model-specific symmetries:
* *$J_1$-$J_2$ Heisenberg chain*
  * spin permutation $S_3$ to prune SDP variables
  * spin rotation $SO(3)$ to generate Ward identities
  * spin $\pi$-rotation subgroup $D_2\cong C_2\times C_2$ to further split PSD blocks
* *Hubbard in 1D and 2D*
  * spin-resolved fermion parity $P_\uparrow\times P_\downarrow$ to further split PSD blocks
  * charge $U(1)$ symmetry to generate Ward identities; fixed total particle number
  * spin rotation $SU(2)$ to generate Ward identities
  * spin-resolved $\pi/2$-rotations $C_{4,\uparrow}\times C_{4,\downarrow}$ in the Majorana planes, i.e. a finite Abelian subgroup of $U_\uparrow(1)\times U_\downarrow(1)$, to prune SDP variables
  * spin exchange to prune SDP variables
  * additional symmetries at half-filling on bipartite lattices:
    * particle-hole symmetry to simplify PSD constraints
    * $\eta$-pairing $SU(2)_\eta$ to generate Ward identities

## Dependencies
* python >= 3.13
* numpy, scipy, scikit-sparse, lru-dict
* cppimport, pybind11, and a C++20 compiler with OpenMP for the JIT modules
* [scs-qmbboot](https://github.com/JefferyWangSH/scs-qmbboot) for fast, memory-efficient SDP solving with warm starts
* (optional) cvxpy with, e.g., MOSEK, SCS as alternative SDP backends

## Usage
The main entry point is [`main.py`](main.py). For example, start an NGA run with:
```bash
python main.py --config config/hubbard_scs.json --output-dir results --steps 10
```
Example configurations are available in [`config/`](config/). For more details, see the [wiki](https://deepwiki.com/JefferyWangSH/QMBBoot-NGA), which also includes a built-in AI agent to play with.

## References
* https://arxiv.org/pdf/2410.00810
* https://arxiv.org/pdf/2310.05844
* https://arxiv.org/pdf/2406.17844
* https://arxiv.org/pdf/2006.06002
* https://arxiv.org/pdf/2111.13007
* https://arxiv.org/pdf/2412.07837
* https://github.com/EverettYou/QuantumBootstrap

## License
MIT

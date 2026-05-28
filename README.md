# QMBBoot-NGA
* *Nullspace-guided adaptive algorithm of quantum many-body bootstrap (QMBBoot-NGA).*
* Currently support computing ground-state energy lower bounds for finite lattice chains.
* Utilize symmetries to
  * expose block structure in PSD moment matrices,
  * prune SDP optimization variables,
  * and impose Ward-identity constraints.
* Use [CVXPY](https://github.com/cvxpy/cvxpy) for solving SDPs.
* Build PSD blocks as vectorized sparse affine maps and affine constraints as a sparse matrix.

## Models
Implemented:
* *Transverse/Longitudinal-field Ising chain*
* *$J_1$-$J_2$ Heisenberg chain*
* *Hubbard chain*

All three model compilers use:
* lattice translation symmetry to generate momentum PSD blocks
* lattice inversion
* complex conjugation $\mathcal{K}$-symmetry

Additional model-specific symmetries:
* *$J_1$-$J_2$ Heisenberg chain*
  * spin label permutation $S_3$
  * spin rotation $SO(3)$; $\pi$-rotation subgroup $C_2^3$ to further split PSD blocks
* *Hubbard chain*
  * charge symmetry $U_\uparrow(1)\times U_\downarrow(1)$; fixed filling number
  * total fermion parity $P$ to further split PSD blocks; consider $P_\uparrow \times P_\downarrow$ as a natural next refinement

## TODOs
* Observables beyond ground-state energy.
* Bootstrap thermal states.

## Refs
* https://arxiv.org/pdf/2410.00810
* https://arxiv.org/pdf/2310.05844
* https://arxiv.org/pdf/2406.17844
* https://arxiv.org/pdf/2006.06002
* https://arxiv.org/pdf/2111.13007
* https://arxiv.org/pdf/2412.07837
* https://github.com/EverettYou/QuantumBootstrap

## License
MIT

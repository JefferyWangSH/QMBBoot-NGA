# QMBBoot-NGA
* *Nullspace-guided adaptive algorithm of quantum many-body bootstrap (QMBBoot-NGA).*
* Support computing lower bounds of ground-state energy density and certified lower/upper bounds of observables for finite lattice chains.
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
* lattice inversion to prune SDP variables
* complex conjugation $\mathcal{K}$-symmetry to prune $\mathcal{K}$-odd SDP variables and identify equivalent $k$ and $-k$ momentum PSD blocks

Additional model-specific symmetries:
* *$J_1$-$J_2$ Heisenberg chain*
  * spin label permutation $S_3$ to prune SDP variables
  * spin rotation $SO(3)$ via Ward identities
  * $\pi$-rotation spin subgroup $D_2=C_2\times C_2$ to further split PSD blocks
* *Hubbard chain*
  * spin-resolved fermion parity $P_\uparrow\times P_\downarrow$ to further split PSD blocks
  * charge symmetry $U(1)$ via Ward identities; fixed total filling number
  * spin rotation $SU(2)$ via Ward identities
  * spin-resolved $\pi/2$ rotations $C_{4,\uparrow}\times C_{4,\downarrow}$ in the Majorana planes, i.e. a finite Abelian subgroup of $U_\uparrow(1)\times U_\downarrow(1)$, to prune SDP variables
  * spin exchange to prune SDP variables

## TODOs
- [ ] Bootstrap thermal states.

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

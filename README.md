# Quantum many-body bootstrap
* Currently support computing ground-state energy lower bounds for finite-size models.
* Utilize symmetries to
  * expose block structure in PSD moment matrices,
  * prune SDP optimization variables,
  * and impose Ward-identity constraints.
* Use [CVXPY](https://github.com/cvxpy/cvxpy) for solving SDPs.
* Build PSD blocks as vectorized sparse affine maps and affine constraints as a sparse matrix.
* See [note.ipynb](note.ipynb) for a concise introduction to the method and benchmark results.

## Models
* *Transverse-field Ising chain*
  * lattice translation symmetry to generate momentum PSD blocks
  * $\mathcal{K}$-symmetry (complex conjugation)
* *$J_1$-$J_2$ Heisenberg chain*
  * lattice translation symmetry to generate momentum PSD blocks
  * lattice inversion
  * $\mathcal{K}$-symmetry (complex conjugation)
  * spin label permutation $S_3$
  * $SO(3)$ spin rotation; $\pi$-rotation subgroup $C_2^3$ to further divide PSD blocks
* *Hubbard chain*
  * lattice translation symmetry to generate momentum PSD blocks
  * $\mathcal{K}$-symmetry (complex conjugation)
  * fermion parity
  * $U_\uparrow(1)\times U_\downarrow(1)$ charge symmetry; fixed filling number

## TODOs
Check https://arxiv.org/abs/2412.07837.
* Observables beyond ground-state energy.
* Bootstrap thermal states.
* Infinite-chain bootstrap based on compatible local reduced density matrices.

## Refs
* https://arxiv.org/pdf/2406.17844
* https://arxiv.org/pdf/2111.13007
* https://github.com/EverettYou/QuantumBootstrap
* https://arxiv.org/pdf/2006.06002
* https://arxiv.org/pdf/2410.00810

## License
MIT

# Quantum many-body bootstrap
* Currently support computing ground-state lower bounds for finite-size models.
* Utilize symmetries to
  * expose block structure in moment PSD matrices,
  * prune optimization variables,
  * and impose Ward-identity constraints.
* Use [CVXPY](https://github.com/cvxpy/cvxpy) for solving SDPs.
* Build PSD blocks as vectorized sparse affine maps and affine constraints as a sparse matrix.

## Models
* *Transverse-field Ising chain*
  * Lattice translation symmetry to generate momentum PSD blocks
  * $\mathcal{K}$-symmetry (complex conjugation)
* *Hubbard chain*
  * Lattice translation symmetry to generate momentum PSD blocks
  * Fermion parity
  * $U_\uparrow(1)\times U_\downarrow(1)$ symmetry; fixed filling number

## TODOs
Check https://arxiv.org/abs/2412.07837.
* Bootstrap thermal states.
* Infinite-chain bootstrap based on compatible local reduced density matrices.

## License
MIT
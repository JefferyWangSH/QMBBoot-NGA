# Quantum many-body bootstrap
* Currently support:
  * Finite-length transverse-field Ising chain
    * Lattice translation symmetry
    * $\mathcal{K}$-symmetry (complex conjugation)
  * Finite-length Hubbard chain
    * Lattice translation symmetry
    * Fermion parity
    * $U_\uparrow(1)\times U_\downarrow(1)$ symmetry; fixed filling number
* Use [cvxpy](https://github.com/cvxpy/cvxpy) for solving SDP.

## TODOs
Check https://arxiv.org/abs/2412.07837.
* Bootstrap thermal states.
* Infinite-chain bootstrap based on compatible local reduced density matrices.
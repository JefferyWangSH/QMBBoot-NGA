import time

import cvxpy as cp

from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
from sdp import SDPSolver, ScsSDPSolver

if __name__ == '__main__':
    params = HubbardParams(L=10, t=1., U=4., n_particles=10)
    compiler = HubbardCompiler(params=params)
    compiler.compile(basis_reprs=build_basis_reprs(
        params.L,
        max_degree=4,
        max_support=2,
        max_diameter=1,
    ))
    print(compiler.summary())
    data = compiler.sdp_data()
    eps = 1e-5

    # same SDPData: cvxpy + SCS (complex blocks real-embedded as 2d x 2d) vs scs-python
    # (native complex PSD cones); realified blocks are real d x d blocks in both;
    # same SCS settings (adaptive_diag_scale is ScsSDPSolver's default, passed explicitly to cvxpy)
    energies = {}
    for name, solver, kwargs in [
        ('cvxpy SCS', SDPSolver(), dict(backend='SCS', eps=eps, adaptive_diag_scale=0)),
        ('native SCS', ScsSDPSolver(), dict(eps=eps)),
    ]:
        solver.build(data)
        if isinstance(solver, SDPSolver):
            n_rows = solver.problem.get_problem_data(cp.SCS)[0]['A'].shape[0]
        else:
            n_rows = solver.problem[0]['A'].shape[0]

        start = time.perf_counter()
        energies[name] = solver.solve(**kwargs)
        wall = time.perf_counter() - start

        summary = solver.summary()
        print(
            f"{name:10s} | rows {n_rows:7d} | {summary['status']:18s} | "
            f"iters {summary['num_iters']:6d} | {wall:7.2f} s | energy {energies[name]:.8f}"
        )

    print(f"|energy diff| = {abs(energies['cvxpy SCS'] - energies['native SCS']):.1e}")

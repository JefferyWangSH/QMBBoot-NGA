from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
from sdp import SDPSolver

if __name__ == '__main__':
    params = HubbardParams(L=16, t=1., U=4., n_particles=16)
    compiler = HubbardCompiler(params=params)
    compiler.compile(basis_reprs=build_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    ))
    print(compiler.summary())

    solver = SDPSolver()
    solver.build(compiler.sdp_data())
    value = solver.solve(backend='SCS', eps=1e-4, max_iters=5_000)
    print(value)
    print(solver.summary())

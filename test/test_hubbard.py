from hubbard.hubbard import HubbardCompiler, HubbardParams, MajoranaMonomial
from sdp import SDPSolver

if __name__ == '__main__':
    params = HubbardParams(L=16, t=1., U=4.)
    compiler = HubbardCompiler(params=params)
    compiler.compile(basis_reprs=[
        MajoranaMonomial.identity(L=params.L),
        MajoranaMonomial.from_str(L=params.L, s='0u+ 1u-'),
        MajoranaMonomial.from_str(L=params.L, s='0u- 1u+'),
        MajoranaMonomial.from_str(L=params.L, s='0d+ 1d-'),
        MajoranaMonomial.from_str(L=params.L, s='0d- 1d+'),
        MajoranaMonomial.from_str(L=params.L, s='0u+ 0u- 0d+ 0d-'),
    ])
    print(compiler.summary())

    solver = SDPSolver()
    solver.build(compiler.sdp_data())
    value = solver.solve(solver='SCS', eps=1e-4, max_iters=5_000)
    print(value / params.L)
    print(solver.summary())

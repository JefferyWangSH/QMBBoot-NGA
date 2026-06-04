from nga import NGAParams, NGARunner
from nga_scheduler import BaseScheduler
from compiler.hubbard import HubbardCompiler, HubbardParams, build_basis_reprs
from operators.majorana import MajoranaMonomial

if __name__ == '__main__':

    params = HubbardParams(L=10, t=1., U=4., n_particles=10)
    basis_reprs = build_basis_reprs(
        params.L,
        max_degree=4,
        max_support=1,
        max_diameter=0,
    )
    required_basis_reprs = [
        MajoranaMonomial.from_str(params.L, monomial).trans_canon_rep
        for monomial in ['I', '0u+', '0u-', '0u+ 0u-', '0d+', '0d-', '0d+ 0d-']
    ]

    runner = NGARunner(
        compiler=HubbardCompiler(params),
        basis_reprs=basis_reprs,
        required_basis_reprs=required_basis_reprs,
        nga_params=NGAParams(
            solver_backend='SCS',
            solver_kwargs={
                'eps': 1e-5,
                'max_iters': 5000
            },
            drop_null_tol=1e-5,
            grow_null_tol=1e-5,
            max_drop_leverage=5e-2,
        ),
        scheduler=BaseScheduler(
            net_growth_min=1,
            net_growth_cap=8,
            drop_cap=8,
            reentry_penalty=0.5,
        ),
    )

    n_steps = 10
    for i in range(n_steps):
        summary, record = runner.step()
        print(i, summary)
        print(i, record)

from nga import NGAParams, NGARunner
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
        MajoranaMonomial.from_str(params.L, monomial)
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
            grow_null_tol=1e-6,
            max_drop_leverage=5e-2,
            min_net_growth_per_step=1,
            max_net_growth_per_step=8,
            drop_cap_base_per_step=8,
            drop_cap_rate=0.15,
        ),
    )

    n_steps = 8
    for i in range(n_steps):
        summary, record = runner.step()
        print(i, summary)
        print(i, record)
